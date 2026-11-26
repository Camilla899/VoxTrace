# Model weights

The two ONNX files in this directory are **required at runtime** but are not
committed to the source repository. The repository and the GitHub web upload
page only accept small files; `rmvpe.onnx` is about 345 MB and exceeds the
25 MB browser upload limit and the 100 MB normal Git limit.

Place these exact files here:

| File | Size | SHA-256 |
|---|---:|---|
| `basic_pitch.onnx` | 230,444 bytes | `2C3C1D144BFA61AD236E92E169C13535C880469A12A047D4E73451F2C059A0EC` |
| `rmvpe.onnx` | 361,688,443 bytes | `5370E71AC80AF8B4B7C793D27EFD51FD8BF962DE3A7EDE0766DAC0BEFA3660FD` |

The ready-to-run portable build already contains both under
`dist/VoxTrace/_internal/models/`; the local development copy is in this
folder on the build machine.

## Sources

- **Basic Pitch** — Spotify Basic Pitch, Apache-2.0. Model/code:
  https://github.com/spotify/basic-pitch
- **RMVPE** — RVC Project / `rmvpe-onnx`, MIT. Weights/algorithm:
  https://github.com/RVC-Project/Retrieval-based-Voice-Conversion
  https://github.com/NewComer00/rmvpe-onnx
  https://huggingface.co/lj1995/VoiceConversionWebUI

The ONNX files may be copied from an existing VoxTrace release. If you rebuild
the RMVPE export yourself, keep the filename `models/rmvpe.onnx`.

## If the models must live in the repository

Use Git LFS with the Git command line (not the GitHub web uploader):

```text
git lfs install
git lfs track "models/*.onnx"
git add .gitattributes
git add -f models/basic_pitch.onnx models/rmvpe.onnx
git commit -m "Add ONNX model weights via Git LFS"
git push
```

`.gitignore` intentionally excludes `models/*.onnx`, hence the `-f` on the
two model files. Without LFS, leave them out of the repository and distribute
them separately.

`rmvpe.onnx` is larger than 100 MB, so a normal `git push` will be rejected;
LFS is required. The app still works offline once the files are present.
