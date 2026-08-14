# PRAA Stage-1 artifact and data-seal receipt

Decision: `PASS_PRAA_STAGE1_ARTIFACT_AND_DATA_SEAL`

- successful workflow: [Run #2](https://github.com/hko920920/SelectLLM-ICLR2026-/actions/runs/31775680237)
- receipt SHA-256: `cf73a7c145565c165f718a059acbca28b36abebd8b906d3ede213ba0c95e666b`
- public artifact ID: `9209815865`
- public artifact SHA-256: `e4a4ca0e61a6ffba59f9ccb5604e9befbcb168f344cf9bc917a3aa1ab9b81a1e`
- separately sealed-label artifact ID: `9209816298`
- sealed artifact SHA-256: `ae4be6bb3d6701123f6558d135d006aeeaa7af2f0e376405659e26ec12b54c57`

The public artifact was inspected. The separately sealed-label artifact was
**not downloaded or opened** in this continuation session.

## Information boundary retained

Stage 1 read exact public rows and labels only to build deterministic,
separately stored input and label packages. It did not:

- load model tensors;
- run any clean or alias endpoint;
- inspect predictions, representations, logits, or confidence;
- calculate candidate accuracy or choose a parent;
- run the selector;
- inspect a primary or deployment scientific outcome.

## Exact data seals

### Primary emotion task

Dataset: `dair-ai/emotion@cab853a1dbdf4c42c2b3ef2173804746df8825fe`

- snapshot: 19 files, 28,186,659 bytes;
- snapshot inventory SHA-256:
  `bf82fa751aa0cb054087b2f4cc3cb75908ff20fcb9f199d78466d2ba9392e334`;
- exact split cardinalities: 16,000 train / 2,000 validation / 2,000 test;
- exact label order: sadness, joy, love, anger, fear, surprise.

| Partition | Rows | UID SHA-256 | Input SHA-256 | Sealed-label SHA-256 |
|---|---:|---|---|---|
| `error_head_train` | 12,000 | `e6bff1230d40f49f5471b9a218cd428700bb64d03482c55812196876c8438508` | `2dc209019663c51db5249e3e0c03df82007465dd5a071454924043daecd493d3` | `8433e21f2bab5352557a24158538170c704c414235a0f8cb7ff3d8d3cc8eca33` |
| `safety` | 4,000 | `50eda8dc3c0c2cbc0d5c1cf44e68e77d0baa96fa8b018beb1dee3317b3575cd0` | `faeaeb364ba47be806bde551f2f365b8b28463e7e9fe5bfdb13bd0c171c1860a` | `e2fbcf13defe650d8b510165f74a0bca881f37425553a390e64ea6805e1986a2` |
| `target_threshold` | 2,000 | `a9a3f318f97257579270e95df866718ceb52dba8c8c0998aa59faacd4c927f7f` | `a5a162b98711179e9e88860cffa0acd242ef62077e64a2aff3296648b57b9db9` | `602afb1d3c59dd9bc22e613e1319760110f32804e074dd2c43f31ab69ad049ed` |
| `primary_outcome` | 1,500 | `43cd1c81140649bc9c1a089fd02cb4ec1b110d4b8aa84675bd955caea9505f12` | `1982a1f00daaf2073df9ea54f35e9336e9cda1664ec17b2fa6e8ea257eaecedd` | `5a0d39660a8ddbc789a697c50308a738a6031129b725d44cfdcf7b9d79f06699` |
| `deployment` | 500 | `9a0a5bb4cf4a2cc221bbac591f76e8641392213ad5192bb56c16be82e822989d` | `7e796963424e7429d4d1fcc5eb154e953742cba5f6b102301d02680a73c0ff29` | `94b46b8470f334cc401aa8297f66af8487d1c428a950f17da1ac3342ac9511d1` |

### Language-identification replication

Dataset: `papluca/language-identification@aa56583bf2bc52b0565770607d6fc3faebecf9e2`

- snapshot: 16 files, 15,349,749 bytes;
- snapshot inventory SHA-256:
  `2c5c277d4a0354ff659cf4097b35c106f9b28dac0581fc341054ef43e2a620e8`;
- exact split cardinalities: 70,000 train / 10,000 validation / 10,000 test;
- exact 20-label set was verified.

| Partition | Rows | UID SHA-256 | Input SHA-256 | Sealed-label SHA-256 |
|---|---:|---|---|---|
| `error_head_train` | 50,000 | `99b7e60b24107c27f26548312cfe8f41709740ba9133076c3ccecb4b1a51f0e1` | `4956909e655b2bd008bb77cf2154f96970e884579e0d5015290dc23560873c22` | `0c32864e26e6e792c18a42eb8736cc6bdccd5538f4bcf1889e38bd8cf5500766` |
| `safety` | 10,000 | `0fc55b9c21985180f3cf7b538900bfab954ab304a4403cb1d703ba7e9c919d07` | `83749644104ccf5176669fb47bb24e44f959447e84ff53607944f120d6c6b078` | `fd3c0d5d654064a14ffd75605a5f086eb79a89b4d37c973d22c2b6a5c4e97c55` |
| `development_reserve` | 10,000 | `58d99aeecd133986371260dff31aa237e1447e034a83e25499561b6c10a21514` | `3800ab35dd173330e13ead168a12c9f288186512ec68b2634ed2c268e5e14501` | `10b0b85335365dab3d83b23e97b75c5c96f017e8435cf457173952a3724de7eb` |
| `target_threshold` | 10,000 | `24cf0e43928eb4a7c962dda55f267ae273270eb891859de3fa581815f78bfba1` | `e74c5b54f9a50883075392a16cc798d7823055ec781f89259341cb7f7c807624` | `6abb1177292080497b887484a5b28b766c7d2641a59a94f40d80af209b1a9c0b` |
| `primary_outcome` | 8,000 | `12c6b0a422b58978d04d29537f9ab0c43fd8394780ba5bed237dd11cb1fe88e5` | `25de0c0077b34f03e345d218049c96b4e45cdc86cfe90bded1892e5b66bf7434` | `5b9314cb0487ed1ec4c3f03d52d9dcf16b957cedf326d9e1130a1e4ef9021af9` |
| `deployment` | 2,000 | `7a060d793eaf7fe61d30ce977dfbbd00300469e7c2d01457114bd94d5f3b6d60` | `4783a7377081439c62d5ce756ebd9085b2a8c0b64462b37a02e2c242e6d04d4a` | `90818793ff281bd2d3bc6ee38225b2fad9857fd33ebeed2de98c46e1e341ff55` |

## Runtime artifact bindings

The Stage-1 lock records the selected runtime weight/package and supporting
configuration/tokenizer hashes for all eight clean roots. The principal weight
bindings are:

- Emotion RoBERTa: `model.safetensors` → `71254cc48a4ac80612a6117060a934152b28926b803cd8c35b01abe8b13b59c7`
- Emotion BERT: `pytorch_model.bin` → `7f656183c3df65dba3b1f3023cd069f1e9b185787891d2c2fea60d8634f058d3`
- Emotion DistilBERT: `model.safetensors` → `a2e46512c3fe95064cc92cd00d7c52a96be25ac6d753ee736e93438f8aa6a09e`
- Emotion ALBERT: `pytorch_model.bin` → `73b6468798f9155ca1710fe47331a04d7dbb6452d5a9f134327fffc154f1516e`
- LID XLM-R: `model.safetensors` → `a835d6e8ed50ef6b3c180db87446a83ee5ac437e981c932c8e1e239aacbe08b7`
- Meta fastText LID: `model.bin` → `8ded5749a2ad79ae9ab7c9190c7c8b97ff20d54ad8b9527ffa50107238fc7f6a`
- GlotLID v3: `model_v3.bin` → `a818b6bd42a628ab47d3dfc1578c7ea615c45381f3494c42535e31e8c4cafc9e`
- `langid==1.1.6`: source distribution → `044bcae1912dab85c33d8e98f2811b8f4ff1213e5e9a9e9510137b84da2cb293`

## Next boundary

A later Stage 2 may run **clean endpoint inference only** after freezing each
wrapper, preprocessing rule, label map, CPU determinism setting, and parent
selection/tie rule. It may use training and safety labels for clean-root
certification. Primary and deployment outcomes remain sealed for scientific
analysis, and no selector is authorized yet.
