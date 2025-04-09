def log_step(epoch, step, g_loss, content_loss, gan_loss, d_loss):
    print(f"[Epoch {epoch} | Step {step}] "
          f"G_total: {g_loss:.4f} | Content: {content_loss:.4f} | "
          f"GAN: {gan_loss:.4f} | D: {d_loss:.4f}")
