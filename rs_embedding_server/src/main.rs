//! Pure-Rust OpenAI-compatible embedding + rerank server on the llama.cpp-HIP base.
//!
//! Lane-A replacement for the Python embed-pool (gateway.py + workers). Byte-compatible with the ATOM
//! `atom.entrypoints.openai.api_server` + gateway contract (see
//! `projects/rust/docs/EMBED-POOL-RUST-MIGRATION-SPEC.md`):
//!   POST /v1/embeddings          — dense, Matryoshka `dimensions`, last-token pooling
//!   POST /v1/rerank              — ColBERT MaxSim (ATOM schema: `top_k` / `score`)
//!   GET  /v1/models              — served ids + aliases
//!   GET  /health                 — gateway aggregate {status, mode, workers{}}
//!   GET  /v1/pool/status         — per-worker lifecycle
//!   POST /v1/pool/{name}/wake    — load a sleeping worker
//!   POST /v1/pool/{name}/sleep   — unload a running worker
//!
//! Supervisor (M6): each model runs in a dedicated OS thread (a llama context is `!Send`) with a
//! WAKE-ON-DEMAND lifecycle — lazy-load on first request, idle-reap after a per-model timeout, unload,
//! park, reload on the next request. Per-worker state is published via atomics for /health. Consul
//! registration stays at the systemd level (ExecStartPost), same as gateway.py. Test port; the live
//! :9207 is only repointed at the M7 gate.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use rs_llama_cpp_core::{LlamaBackend, LlamaContext, LlamaModel, Pooling};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::oneshot;

// ── Per-model config + registry ───────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum ModelKind {
    Dense,
    ColBert,
}

#[derive(Clone)]
struct ModelConfig {
    model_path: String,
    model_id: String,
    aliases: Vec<String>,
    default_dim: usize,
    allowed_dims: Vec<usize>,
    n_ctx: u32,
    ngl: i32,
    pooling: Pooling,
    kind: ModelKind,
    device: &'static str,
    idle_secs: u64,
}

impl ModelConfig {
    fn all_ids(&self) -> Vec<String> {
        let mut v = vec![self.model_id.clone()];
        v.extend(self.aliases.iter().cloned());
        v
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}
fn dims(s: &str) -> Vec<usize> {
    s.split(',').filter_map(|x| x.trim().parse::<usize>().ok()).collect()
}
fn aliases(s: &str) -> Vec<String> {
    s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()
}

fn default_registry() -> Vec<ModelConfig> {
    let gpu_idle: u64 = env_or("GPU_IDLE_TIMEOUT_SEC", "300").parse().unwrap_or(300);
    let mut reg = vec![
        ModelConfig {
            model_path: env_or(
                "EMBED_QWEN_MODEL",
                "/home/local/ai/models/registry/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf",
            ),
            model_id: "embed-qwen3".into(),
            aliases: aliases("qwen3,qwen3-embedding-0.6b,Qwen/Qwen3-Embedding-0.6B"),
            default_dim: 1024,
            allowed_dims: dims("128,384,512,768,1024"),
            n_ctx: 512,
            ngl: 999,
            pooling: Pooling::Last,
            kind: ModelKind::Dense,
            device: "gpu",
            idle_secs: gpu_idle,
        },
        ModelConfig {
            model_path: env_or(
                "EMBED_JINA_MODEL",
                "/home/local/ai/models/registry/JinaAI/jina-code-embeddings-0.5b/jina-code-0.5b-f16.gguf",
            ),
            model_id: "embed-jina-code".into(),
            aliases: aliases("jina-code,jina-code-embeddings-0.5b,jinaai/jina-code-embeddings-0.5b"),
            default_dim: 896,
            allowed_dims: dims("64,128,256,512,896"),
            n_ctx: 512,
            ngl: 999,
            pooling: Pooling::Last,
            kind: ModelKind::Dense,
            device: "gpu",
            idle_secs: gpu_idle,
        },
    ];
    if env_or("EMBED_ENABLE_4B", "1") != "0" {
        reg.push(ModelConfig {
            model_path: env_or(
                "EMBED_QWEN4B_MODEL",
                "/home/local/ai/models/registry/Qwen/Qwen3-Embedding-4B/quantizations/GGUF/Qwen3-Embedding-4B-Q8_0.gguf",
            ),
            model_id: "embed-qwen3-4b".into(),
            aliases: aliases("qwen3-embedding-4b,Qwen/Qwen3-Embedding-4B"),
            default_dim: 2560,
            allowed_dims: dims("256,512,1024,1536,2560"),
            n_ctx: 512,
            ngl: 999,
            pooling: Pooling::Last,
            kind: ModelKind::Dense,
            device: "gpu",
            idle_secs: gpu_idle,
        });
    }
    if env_or("EMBED_ENABLE_COLBERT", "1") != "0" {
        reg.push(ModelConfig {
            model_path: env_or(
                "EMBED_COLBERT_MODEL",
                "/home/local/ai/models/registry/LiquidAI/LFM2-ColBERT-350M/lfm2-colbert-350m-f16.gguf",
            ),
            model_id: "embed-lfm2-colbert".into(),
            aliases: aliases("lfm2-colbert,lfm2-colbert-350m,LiquidAI/LFM2-ColBERT-350M"),
            default_dim: 0,
            allowed_dims: vec![],
            n_ctx: 1024,
            ngl: 999,
            pooling: Pooling::None,
            kind: ModelKind::ColBert,
            device: "gpu",
            idle_secs: env_or("CPU_IDLE_TIMEOUT_SEC", "600").parse().unwrap_or(600),
        });
    }
    reg
}

const Q_PREFIX: &str = "[Q] ";
const D_PREFIX: &str = "[D] ";
const Q_LEN: usize = 32;
const D_LEN: usize = 512;

// ── Worker status (atomics, read by HTTP handlers) ────────────────────

const ST_STOPPED: u8 = 0;
const ST_LOADING: u8 = 1;
const ST_RUNNING: u8 = 2;

struct WorkerStatus {
    state: AtomicU8,
    last_request_ms: AtomicU64,
    request_count: AtomicU64,
    total_starts: AtomicU64,
}
impl WorkerStatus {
    fn new() -> Self {
        WorkerStatus {
            state: AtomicU8::new(ST_STOPPED),
            last_request_ms: AtomicU64::new(0),
            request_count: AtomicU64::new(0),
            total_starts: AtomicU64::new(0),
        }
    }
    fn set(&self, s: u8) {
        self.state.store(s, Ordering::SeqCst);
    }
    fn state_str(&self) -> &'static str {
        match self.state.load(Ordering::SeqCst) {
            ST_LOADING => "loading",
            ST_RUNNING => "running",
            _ => "stopped",
        }
    }
    fn touch(&self) {
        self.last_request_ms.store(now_ms(), Ordering::SeqCst);
        self.request_count.fetch_add(1, Ordering::SeqCst);
    }
    fn idle_seconds(&self) -> u64 {
        let last = self.last_request_ms.load(Ordering::SeqCst);
        if last == 0 {
            0
        } else {
            now_ms().saturating_sub(last) / 1000
        }
    }
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}
fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

// ── Worker actor ──────────────────────────────────────────────────────

struct EmbedOutput {
    embeddings: Vec<Vec<f32>>,
    total_tokens: usize,
}
struct RerankOutput {
    scores: Vec<f32>,
    tokens_evaluated: usize,
}

enum Job {
    Embed {
        texts: Vec<String>,
        dim: Option<usize>,
        reply: oneshot::Sender<Result<EmbedOutput, String>>,
    },
    Rerank {
        query: String,
        documents: Vec<String>,
        reply: oneshot::Sender<Result<RerankOutput, String>>,
    },
    /// Load the model if asleep, reply when ready (eager warmup / explicit wake).
    Warmup {
        reply: oneshot::Sender<Result<(), String>>,
    },
    /// Unload the model now (explicit sleep).
    Sleep {
        reply: oneshot::Sender<Result<(), String>>,
    },
}

fn spawn_worker(cfg: ModelConfig) -> (mpsc::Sender<Job>, Arc<WorkerStatus>) {
    let (tx, rx) = mpsc::channel::<Job>();
    let status = Arc::new(WorkerStatus::new());
    let st = status.clone();
    let _ = std::thread::Builder::new()
        .name(format!("embed-{}", cfg.model_id))
        .spawn(move || worker_loop(cfg, rx, st));
    (tx, status)
}

/// Wake-on-demand lifecycle: park until a job arrives, load the model, serve until idle, unload, repeat.
fn worker_loop(cfg: ModelConfig, rx: mpsc::Receiver<Job>, status: Arc<WorkerStatus>) {
    let backend = LlamaBackend::init();
    let idle = Duration::from_secs(cfg.idle_secs.max(1));
    loop {
        // Parked (model unloaded). A closed channel means the server is shutting down.
        let first = match rx.recv() {
            Ok(j) => j,
            Err(_) => return,
        };
        // A Sleep arriving while already stopped is a no-op.
        if let Job::Sleep { reply } = first {
            let _ = reply.send(Ok(()));
            continue;
        }

        status.set(ST_LOADING);
        status.total_starts.fetch_add(1, Ordering::SeqCst);
        let model = match LlamaModel::load(&backend, &cfg.model_path, cfg.ngl) {
            Ok(m) => m,
            Err(e) => {
                status.set(ST_STOPPED);
                fail_job(first, format!("model load failed ({}): {e}", cfg.model_path));
                continue;
            }
        };
        let mut ctx: LlamaContext<'_> = match model.embedding_context(cfg.n_ctx, cfg.pooling) {
            Ok(c) => c,
            Err(e) => {
                status.set(ST_STOPPED);
                fail_job(first, format!("context init failed: {e}"));
                continue;
            }
        };
        status.set(ST_RUNNING);

        let mut unload = handle_job(&mut ctx, &status, first);
        while !unload {
            match rx.recv_timeout(idle) {
                Ok(j) => unload = handle_job(&mut ctx, &status, j),
                Err(mpsc::RecvTimeoutError::Timeout) => break, // idle → unload
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            }
        }
        drop(ctx);
        drop(model);
        status.set(ST_STOPPED);
    }
}

/// Returns true if the worker should unload (sleep) after this job.
fn handle_job(ctx: &mut LlamaContext<'_>, status: &WorkerStatus, job: Job) -> bool {
    match job {
        Job::Embed { texts, dim, reply } => {
            status.touch();
            let _ = reply.send(run_embed(ctx, &texts, dim));
            false
        }
        Job::Rerank { query, documents, reply } => {
            status.touch();
            let _ = reply.send(run_rerank(ctx, &query, &documents));
            false
        }
        Job::Warmup { reply } => {
            // Model is loaded by the time we get here.
            let _ = reply.send(Ok(()));
            false
        }
        Job::Sleep { reply } => {
            let _ = reply.send(Ok(()));
            true
        }
    }
}

/// Deliver a load failure to whatever job triggered the (failed) wake.
fn fail_job(job: Job, msg: String) {
    match job {
        Job::Embed { reply, .. } => {
            let _ = reply.send(Err(msg));
        }
        Job::Rerank { reply, .. } => {
            let _ = reply.send(Err(msg));
        }
        Job::Warmup { reply } => {
            let _ = reply.send(Err(msg));
        }
        Job::Sleep { reply } => {
            let _ = reply.send(Ok(()));
        }
    }
}

fn run_embed(ctx: &mut LlamaContext<'_>, texts: &[String], dim: Option<usize>) -> Result<EmbedOutput, String> {
    let mut embeddings = Vec::with_capacity(texts.len());
    let mut total_tokens = 0usize;
    for t in texts {
        total_tokens += ctx.token_count(t).map_err(|e| e.to_string())?;
        embeddings.push(ctx.embed_dim(t, dim).map_err(|e| e.to_string())?);
    }
    Ok(EmbedOutput { embeddings, total_tokens })
}

fn run_rerank(ctx: &mut LlamaContext<'_>, query: &str, documents: &[String]) -> Result<RerankOutput, String> {
    let q = ctx
        .colbert_tokens(&format!("{Q_PREFIX}{query}"), Some(Q_LEN))
        .map_err(|e| e.to_string())?;
    let mut tokens_evaluated = q.len();
    let mut scores = Vec::with_capacity(documents.len());
    for doc in documents {
        let d = ctx
            .colbert_tokens(&format!("{D_PREFIX}{doc}"), Some(D_LEN))
            .map_err(|e| e.to_string())?;
        tokens_evaluated += d.len();
        scores.push(maxsim(&q, &d));
    }
    Ok(RerankOutput { scores, tokens_evaluated })
}

/// ColBERT MaxSim: per-token rows are L2-normalized (dot == cosine); for each query token take the max
/// over doc tokens, then SUM over query tokens — RAW sum (unbounded, not length-normalized) = ATOM.
fn maxsim(query: &[Vec<f32>], doc: &[Vec<f32>]) -> f32 {
    let mut score = 0.0f32;
    for q in query {
        let mut best = f32::NEG_INFINITY;
        for d in doc {
            let dot: f32 = q.iter().zip(d).map(|(a, b)| a * b).sum();
            if dot > best {
                best = dot;
            }
        }
        if best.is_finite() {
            score += best;
        }
    }
    score
}

// ── App state ─────────────────────────────────────────────────────────

struct WorkerHandle {
    tx: mpsc::Sender<Job>,
    cfg: ModelConfig,
    status: Arc<WorkerStatus>,
}

#[derive(Clone)]
struct AppState {
    workers: Arc<Vec<WorkerHandle>>,
    embed_routes: Arc<HashMap<String, usize>>, // dense id/alias -> worker idx
    by_name: Arc<HashMap<String, usize>>,       // model_id -> worker idx (pool control)
    colbert_idx: Option<usize>,
    /// Upper bound on a single request (incl. cold-start load). A hung GPU call must not hang the
    /// client forever — mirrors the Python worker's request timeout. Env `WORKER_TIMEOUT_SEC`.
    request_timeout: Duration,
}

// ── Shared wire helpers ───────────────────────────────────────────────

fn err(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(json!({ "detail": msg.into() }))).into_response()
}
fn default_true() -> bool {
    true
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}

// ── /v1/embeddings ────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(untagged)]
enum Input {
    One(String),
    Many(Vec<String>),
}
impl Input {
    fn into_vec(self) -> Vec<String> {
        match self {
            Input::One(s) => vec![s],
            Input::Many(v) => v,
        }
    }
}

#[derive(Deserialize)]
struct EmbeddingsRequest {
    #[serde(default)]
    model: Option<String>,
    input: Input,
    #[serde(default)]
    dimensions: Option<usize>,
    #[serde(default)]
    #[allow(dead_code)]
    encoding_format: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    user: Option<String>,
}

#[derive(Serialize)]
struct EmbeddingObject {
    object: &'static str,
    index: usize,
    embedding: Vec<f32>,
}
#[derive(Serialize)]
struct EmbeddingsResponse {
    object: &'static str,
    created: u64,
    model: String,
    data: Vec<EmbeddingObject>,
    usage: Usage,
}

async fn embeddings(State(st): State<AppState>, Json(req): Json<EmbeddingsRequest>) -> Response {
    let idx = match req.model.as_deref().filter(|m| !m.is_empty()).and_then(|m| st.embed_routes.get(m).copied()) {
        Some(i) => i,
        None => {
            return err(
                StatusCode::BAD_REQUEST,
                format!(
                    "Unknown embedding model '{}'. Use /v1/models for available ids.",
                    req.model.as_deref().unwrap_or("")
                ),
            )
        }
    };
    let w = &st.workers[idx];

    let texts = req.input.into_vec();
    if texts.is_empty() {
        return err(StatusCode::BAD_REQUEST, "input must not be empty");
    }
    let effective = req.dimensions.unwrap_or(w.cfg.default_dim);
    if effective == 0 {
        return err(StatusCode::BAD_REQUEST, "dimensions must be positive");
    }
    if !w.cfg.allowed_dims.is_empty() && !w.cfg.allowed_dims.contains(&effective) {
        let mut allowed = w.cfg.allowed_dims.clone();
        allowed.sort_unstable();
        let list = allowed.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ");
        return err(
            StatusCode::BAD_REQUEST,
            format!("dimensions={effective} is not supported; allowed dimensions: {list}"),
        );
    }

    let (reply_tx, reply_rx) = oneshot::channel();
    if w.tx.send(Job::Embed { texts, dim: Some(effective), reply: reply_tx }).is_err() {
        return err(StatusCode::SERVICE_UNAVAILABLE, "embedding worker is unavailable");
    }
    let out = match tokio::time::timeout(st.request_timeout, reply_rx).await {
        Ok(Ok(Ok(o))) => o,
        Ok(Ok(Err(e))) => return err(StatusCode::INTERNAL_SERVER_ERROR, e),
        Ok(Err(_)) => return err(StatusCode::SERVICE_UNAVAILABLE, "embedding worker dropped the request"),
        Err(_) => return err(StatusCode::GATEWAY_TIMEOUT, "embedding request timed out"),
    };
    let data = out
        .embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbeddingObject { object: "embedding", index, embedding })
        .collect();
    Json(EmbeddingsResponse {
        object: "list",
        created: now_unix(),
        model: w.cfg.model_id.clone(),
        data,
        usage: Usage { prompt_tokens: out.total_tokens, total_tokens: out.total_tokens },
    })
    .into_response()
}

// ── /v1/rerank (ColBERT) ──────────────────────────────────────────────

#[derive(Deserialize)]
struct RerankRequest {
    #[serde(default)]
    #[allow(dead_code)]
    model: Option<String>,
    query: String,
    documents: Vec<String>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default = "default_true")]
    return_documents: bool,
    #[serde(default)]
    rid: Option<Value>,
    #[serde(default)]
    #[allow(dead_code)]
    user: Option<String>,
}

#[derive(Serialize)]
struct RerankResult {
    index: usize,
    score: f32,
    document: Option<String>,
    meta_info: Option<Value>,
}
#[derive(Serialize)]
struct RerankResponse {
    object: &'static str,
    created: u64,
    model: String,
    id: Option<Value>,
    results: Vec<RerankResult>,
    usage: Usage,
    tokens_evaluated: usize,
}

async fn rerank(State(st): State<AppState>, Json(req): Json<RerankRequest>) -> Response {
    let idx = match st.colbert_idx {
        Some(i) => i,
        None => return err(StatusCode::BAD_REQUEST, "no rerank (ColBERT) model is loaded"),
    };
    let w = &st.workers[idx];

    if req.query.trim().is_empty() {
        return err(StatusCode::BAD_REQUEST, "query must not be empty");
    }
    if req.documents.is_empty() {
        return err(StatusCode::BAD_REQUEST, "documents must not be empty");
    }
    if let Some(k) = req.top_k {
        if k < 1 {
            return err(StatusCode::BAD_REQUEST, "top_n must be >= 1");
        }
    }

    let documents = req.documents.clone();
    let (reply_tx, reply_rx) = oneshot::channel();
    if w.tx.send(Job::Rerank { query: req.query.clone(), documents: documents.clone(), reply: reply_tx }).is_err() {
        return err(StatusCode::SERVICE_UNAVAILABLE, "rerank worker is unavailable");
    }
    let out = match tokio::time::timeout(st.request_timeout, reply_rx).await {
        Ok(Ok(Ok(o))) => o,
        Ok(Ok(Err(e))) => return err(StatusCode::INTERNAL_SERVER_ERROR, e),
        Ok(Err(_)) => return err(StatusCode::SERVICE_UNAVAILABLE, "rerank worker dropped the request"),
        Err(_) => return err(StatusCode::GATEWAY_TIMEOUT, "rerank request timed out"),
    };

    let mut order: Vec<usize> = (0..documents.len()).collect();
    order.sort_by(|&a, &b| {
        out.scores[b]
            .partial_cmp(&out.scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    if let Some(k) = req.top_k {
        order.truncate(k.min(order.len()));
    }
    let results = order
        .into_iter()
        .map(|i| RerankResult {
            index: i,
            score: out.scores[i],
            document: if req.return_documents { Some(documents[i].clone()) } else { None },
            meta_info: None,
        })
        .collect();

    Json(RerankResponse {
        object: "rerank",
        created: now_unix(),
        model: w.cfg.model_id.clone(),
        id: req.rid,
        results,
        usage: Usage { prompt_tokens: out.tokens_evaluated, total_tokens: out.tokens_evaluated },
        tokens_evaluated: out.tokens_evaluated,
    })
    .into_response()
}

// ── pool-wide endpoints ───────────────────────────────────────────────

fn worker_status_json(w: &WorkerHandle) -> Value {
    json!({
        "state": w.status.state_str(),
        "device": w.cfg.device,
        "idle_seconds": w.status.idle_seconds(),
        "idle_timeout": w.cfg.idle_secs,
        "request_count": w.status.request_count.load(Ordering::SeqCst),
        "total_starts": w.status.total_starts.load(Ordering::SeqCst),
    })
}

/// Gateway-shape aggregate health (always HTTP 200 so Consul's check passes; the body is for monitoring).
async fn pool_health(State(st): State<AppState>) -> Json<Value> {
    let mut workers = serde_json::Map::new();
    let mut running = 0usize;
    for w in st.workers.iter() {
        if w.status.state_str() == "running" {
            running += 1;
        }
        workers.insert(w.cfg.model_id.clone(), worker_status_json(w));
    }
    let total = st.workers.len();
    let status = if running == total {
        "ok"
    } else if running > 0 {
        "degraded"
    } else {
        "idle"
    };
    let mode = if running > 0 { "active" } else { "idle" };
    Json(json!({ "status": status, "mode": mode, "workers": workers }))
}

async fn pool_status(State(st): State<AppState>) -> Json<Value> {
    let mut workers = serde_json::Map::new();
    for w in st.workers.iter() {
        workers.insert(w.cfg.model_id.clone(), worker_status_json(w));
    }
    Json(json!({ "workers": workers }))
}

async fn models(State(st): State<AppState>) -> Json<Value> {
    let created = now_unix();
    let mut data = Vec::new();
    for w in st.workers.iter() {
        for id in w.cfg.all_ids() {
            data.push(json!({ "id": id, "object": "model", "created": created, "owned_by": "atom" }));
        }
    }
    Json(json!({ "object": "list", "data": data }))
}

async fn wake_worker(State(st): State<AppState>, Path(name): Path<String>) -> Response {
    let idx = match st.by_name.get(&name) {
        Some(i) => *i,
        None => return err(StatusCode::NOT_FOUND, format!("Unknown worker: {name}")),
    };
    let w = &st.workers[idx];
    let (tx, rx) = oneshot::channel();
    if w.tx.send(Job::Warmup { reply: tx }).is_err() {
        return err(StatusCode::SERVICE_UNAVAILABLE, "worker unavailable");
    }
    match rx.await {
        Ok(Ok(())) => Json(json!({ "status": "ok", "worker": name, "state": w.status.state_str() })).into_response(),
        Ok(Err(e)) => err(StatusCode::SERVICE_UNAVAILABLE, e),
        Err(_) => err(StatusCode::SERVICE_UNAVAILABLE, "worker dropped the wake request"),
    }
}

async fn sleep_worker(State(st): State<AppState>, Path(name): Path<String>) -> Response {
    let idx = match st.by_name.get(&name) {
        Some(i) => *i,
        None => return err(StatusCode::NOT_FOUND, format!("Unknown worker: {name}")),
    };
    let w = &st.workers[idx];
    let (tx, rx) = oneshot::channel();
    if w.tx.send(Job::Sleep { reply: tx }).is_err() {
        return err(StatusCode::SERVICE_UNAVAILABLE, "worker unavailable");
    }
    let _ = rx.await;
    Json(json!({ "status": "ok", "worker": name, "state": "stopped" })).into_response()
}

// ── Main ──────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let registry = default_registry();
    let port: u16 = env_or("EMBED_PORT", "9307").parse().unwrap_or(9307);
    let host = env_or("EMBED_HOST", "127.0.0.1");

    // Spawn all workers (parked, models NOT loaded). Routing is by config, so no load needed here.
    let mut workers = Vec::new();
    let mut embed_routes: HashMap<String, usize> = HashMap::new();
    let mut by_name: HashMap<String, usize> = HashMap::new();
    let mut colbert_idx = None;
    for cfg in registry {
        let idx = workers.len();
        match cfg.kind {
            ModelKind::Dense => {
                for id in cfg.all_ids() {
                    embed_routes.insert(id, idx);
                }
            }
            ModelKind::ColBert => colbert_idx = Some(idx),
        }
        by_name.insert(cfg.model_id.clone(), idx);
        let (tx, status) = spawn_worker(cfg.clone());
        tracing::info!(id = %cfg.model_id, device = cfg.device, idle_secs = cfg.idle_secs, "worker spawned (wake-on-demand)");
        workers.push(WorkerHandle { tx, cfg, status });
    }

    let state = AppState {
        workers: Arc::new(workers),
        embed_routes: Arc::new(embed_routes),
        by_name: Arc::new(by_name),
        colbert_idx,
        request_timeout: Duration::from_secs(env_or("WORKER_TIMEOUT_SEC", "300").parse().unwrap_or(300)),
    };

    // Eager-warm a configured set in the background (default mirrors gateway.py EAGER_START: the
    // light models, NOT the 4b). Empty string disables. The server starts serving immediately.
    let eager = env_or("EMBED_EAGER", "embed-qwen3,embed-jina-code,embed-lfm2-colbert");
    {
        let state = state.clone();
        tokio::spawn(async move {
            for name in eager.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()) {
                if let Some(&idx) = state.by_name.get(&name) {
                    let (tx, rx) = oneshot::channel();
                    if state.workers[idx].tx.send(Job::Warmup { reply: tx }).is_ok() {
                        match rx.await {
                            Ok(Ok(())) => tracing::info!(model = %name, "eager-warmed"),
                            Ok(Err(e)) => tracing::error!(model = %name, "eager warm failed: {e}"),
                            Err(_) => {}
                        }
                    }
                }
            }
        });
    }

    let app = Router::new()
        // /health is the GATEWAY aggregate (what Consul checks + what :9207 returns today).
        .route("/health", get(pool_health))
        .route("/v1/models", get(models))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/rerank", post(rerank))
        .route("/v1/pool/status", get(pool_status))
        .route("/v1/pool/{name}/wake", post(wake_worker))
        .route("/v1/pool/{name}/sleep", post(sleep_worker))
        .with_state(state);

    let addr = format!("{host}:{port}");
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("bind {addr} failed: {e}");
            std::process::exit(1);
        }
    };
    tracing::info!("rs-embedding-server listening on http://{addr}");
    if let Err(e) = axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
    {
        tracing::error!("server error: {e}");
        std::process::exit(1);
    }
}
