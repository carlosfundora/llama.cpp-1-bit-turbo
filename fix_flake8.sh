sed -i 's/print(/logging.info(/g' scripts/spec_harness.py
sed -i 's/print(/logging.info(/g' cmake/triton_aot_compile.py
sed -i 's/n_embd = .*//g' scripts/spec_harness.py
sed -i "s/f'Found /'Found /g" scripts/spec_harness.py
sed -i "s/f'Finalizing /'Finalizing /g" scripts/spec_harness.py
sed -i "s/f'Saving /'Saving /g" scripts/spec_harness.py
sed -i "s/f'Created /'Created /g" scripts/spec_harness.py
sed -i "s/f'Finished /'Finished /g" cmake/triton_aot_compile.py
