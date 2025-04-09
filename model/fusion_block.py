from model.fusion_block import cbam_block, transformer_block

# 在 bottleneck 中添加 CBAM 或 Transformer
c4 = Conv2D(256, 3, activation='relu', padding='same')(p3)
c4 = res_block(c4, 256)
c4 = cbam_block(c4)  # 或 transformer_block(c4)
