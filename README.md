# volume-prediction (temporary)

## TimeGAN Implementation (Dropped)
bash run_TimeGAN.sh

## Adversarial Sparse Transformer Implementation
bash run_AST.sh

(Changed Loss to WGAN-GP loss)


## Autoformer Implementation
bash run_Autoformer.sh

If you want to change the model (Autoformer) into Reformer or Informer, change the ['arch_G']['type'] paramer to "Informer" or "Reformer" in ./config/config_Autoformer.json
