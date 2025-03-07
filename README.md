# Deep Convolutional GAN (DCGAN) for Image Generation

## Dataset Preprocessing
1. **Load the Dataset**: The dataset consists of 31,000 images, loaded using a PyTorch `DataLoader`.
2. **Resize & Normalize**: Images are resized and normalized to the range [-1, 1] for stable GAN training.
3. **Batching**: The data is loaded in batches to optimize training performance (batch size can be adjusted based on system resources).

## Model Training
### Steps to Train:
1. **Initialize Models**: 
   - `netD`: Discriminator model that classifies real and fake images.
   - `netG`: Generator model that creates synthetic images from random noise.
2. **Training Loop**:
   - **Step 1**: Train Discriminator (`netD`):
     - Compute loss on real and fake images.
     - Update `netD` using backpropagation.
   - **Step 2**: Train Generator (`netG`):
     - Generate fake images.
     - Compute loss based on Discriminator’s predictions.
     - Update `netG` using backpropagation.
3. **Logging**:
   - Loss values for Discriminator (`LossD`) and Generator (`LossG`) are printed every 100 batches.
   - Sample generated images are saved after each epoch.

### Command to Run Training:
```bash
python train.py
```

## Model Testing
### Steps to Test:
1. Load the trained Generator (`netG`).
2. Generate fake images from random noise.
3. Evaluate the quality using visual inspection and pixel distribution analysis.

### Expected Outputs:
- **Loss Trends:**
  - `LossD` should stabilize around a value where it can distinguish real and fake images effectively.
  - `LossG` should gradually decrease, indicating improvement in generated images.
- **Discriminator Accuracy:**
  - During training, it should stabilize around 80-90%.
- **Generated Image Quality:**
  - Initially, images may be noisy but should improve over epochs.
  - The final output should resemble real images from the dataset.

## Sample Output
- **Training Log:**
  ```
  [Epoch 0/1] [Batch 0/244] LossD: 0.5325, LossG: 2.2225
  [Epoch 0/1] [Batch 100/244] LossD: 0.4134, LossG: 3.8799
  [Epoch 0/1] [Batch 200/244] LossD: 0.4391, LossG: 2.6475
  Training Completed!
  Discriminator Accuracy: 83.93%
  ```
- **Generator Output Distribution:**
  ![Generator Output](output.png)
