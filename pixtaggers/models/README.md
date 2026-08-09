**Put your model and metadata to this directory**

## Where to download models?

The configured model is selected with `model` in `config.json`:

- `camie-tagger-v2`
- `cl-tagger-v2`

### CamieTagger V2

Get `camie-tagger-v2.onnx` and `camie-tagger-v2-metadata.json` from [CamieTagger V2 on Hugging Face](https://huggingface.co/Camais03/camie-tagger-v2/tree/main).

Save them to `./camie-tagger-v2`.

### CL Tagger V2

Get the following files from [CL Tagger V2 on Hugging Face](https://huggingface.co/cella110n/cl_tagger_v2) and save them to `./cl-tagger-v2`:

I recommend using the `v2_01a` version.

- `model.onnx`
- `model.onnx.data`
- `model_metadata.json`
- `model_vocabulary.json`
- `model_tag_metrics.npz`
- `model_ood_ref.npz`

The repository requires accepting its access conditions before downloading.
