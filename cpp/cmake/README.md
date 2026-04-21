The project now supports both:
- legacy `cpp/Makefile`
- modern `cpp/CMakeLists.txt`

Python training orchestration lives outside this folder under
`src/minicnn/training/`; see the repository README and `docs/guide_project_structure.md`
for the current trainer module map.

Current native API notes:
- both cuBLAS and handmade variants are expected to export the same required symbols
- `maxpool_backward_nchw_status` returns a CUDA status code for Python wrappers that need a catchable error path
- `MINICNN_DEBUG_SYNC` should be enabled only for debug builds that need synchronous kernel checks
