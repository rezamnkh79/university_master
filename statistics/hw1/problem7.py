import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

print("generating picture waiting ........")
img = cv2.imread('cookie_monster.jpg', 0)

noise_ratios = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

fig, axs = plt.subplots(3, 6, figsize=(15, 10))

for i, noise_ratio in enumerate(noise_ratios):
    noisy_img = img.copy()
    row, col = noisy_img.shape
    number_of_pixels = int((noise_ratio * row * col) / 2)
    coords = np.random.randint(0, row, size=(number_of_pixels, 2))
    rows = coords[:, 0]
    cols = coords[:, 1]
    noisy_img[rows, cols] = 255
    coords = np.random.randint(0, row, size=(number_of_pixels, 2))
    rows = coords[:, 0]
    cols = coords[:, 1]
    noisy_img[rows, cols] = 0
    axs[0, i].imshow(noisy_img, cmap='gray')
    axs[0, i].set_title(f'Uniform - {noise_ratio * 100}%')
    axs[0, i].axis('off')
for i, noise_ratio in enumerate(noise_ratios):
    mean = np.random.uniform(0, 255)
    std = np.random.uniform(10, 50) * noise_ratio
    noise = np.random.normal(mean, std, img.shape)
    noisy_img = img.copy()
    noisy_img = noisy_img + noise
    axs[1, i].imshow(noisy_img, cmap='gray')
    axs[1, i].set_title(f'Gaussian - {noise_ratio * 100}%')
    axs[1, i].axis('off')

for i, noise_ratio in enumerate(noise_ratios):
    noise = np.random.exponential(scale=1, size=img.shape) * noise_ratio * 10
    noisy_img = img.copy()
    noisy_img = noisy_img + noise
    axs[2, i].imshow(noisy_img, cmap='gray')
    axs[2, i].set_title(f'Exponential - {noise_ratio * 100}%')
    axs[2, i].axis('off')

plt.show()

print("part A finished.")
print("=" * 150)
print("Part B started")


def count_corrupted_pixels(original_face, noisy_face, threshold=0.2):
    diff = np.abs(original_face - noisy_face)
    corrupted_pixels = np.sum(diff > 20)
    return corrupted_pixels / original_face.size >= threshold


face_pos = (0, 0, 1000, 470)
original_face = img[face_pos[1]:face_pos[3], face_pos[0]:face_pos[2]]

noise_ratios = [0.2, 0.3, 0.4, 0.5]
results = {
    "uniform": [], "gaussian": [], "exponential": []
}

for j, noise_ratio in enumerate(noise_ratios):
    count = 0
    for _ in range(100):
        noisy_img = img.copy()
        row, col = noisy_img.shape
        number_of_pixels = int((noise_ratio * row * col) / 2)
        coords = np.random.randint(0, row, size=(number_of_pixels, 2))
        rows = coords[:, 0]
        cols = coords[:, 1]
        noisy_img[rows, cols] = 255
        coords = np.random.randint(0, row, size=(number_of_pixels, 2))
        rows = coords[:, 0]
        cols = coords[:, 1]
        noisy_img[rows, cols] = 0
        noisy_face = noisy_img[face_pos[1]:face_pos[3], face_pos[0]:face_pos[2]]
        if count_corrupted_pixels(original_face, noisy_face):
            count += 1
    results["uniform"].append(count / 100)

for j, noise_ratio in enumerate(noise_ratios):
    count = 0
    for _ in range(100):
        mean = np.random.uniform(0, 255)
        std = np.random.uniform(10, 50) * noise_ratio
        noise = np.random.normal(mean, std, img.shape)
        noisy_img = img + noise
        noisy_face = noisy_img[face_pos[1]:face_pos[3], face_pos[0]:face_pos[2]]
        if count_corrupted_pixels(original_face, noisy_face):
            count += 1
    results["gaussian"].append(count / 100)

for j, noise_ratio in enumerate(noise_ratios):
    count = 0
    for _ in range(100):
        noise = np.random.exponential(scale=1, size=img.shape) * noise_ratio * 200
        noisy_img = img + noise
        noisy_img = np.clip(noisy_img, 0, 255)
        noisy_face = noisy_img[face_pos[1]:face_pos[3], face_pos[0]:face_pos[2]]
        if count_corrupted_pixels(original_face, noisy_face):
            count += 1
    results["exponential"].append(count / 100)

for noise_type, averages in results.items():
    for noise_ratio, avg_count in zip(noise_ratios, averages):
        print(f'Average corrupted faces for {noise_type} at {noise_ratio * 100}%: {avg_count * 100}')

print("part C")
print("generating picture")
fig, axs = plt.subplots(3, 4, figsize=(15, 10))

for i, noise_ratio in enumerate(noise_ratios):
    noisy_img = img.copy()
    row, col = noisy_img.shape
    number_of_pixels = int((noise_ratio * row * col) / 2)
    coords = np.random.randint(0, row, size=(number_of_pixels, 2))
    rows = coords[:, 0]
    cols = coords[:, 1]
    noisy_img[rows, cols] = 255
    coords = np.random.randint(0, row, size=(number_of_pixels, 2))
    rows = coords[:, 0]
    cols = coords[:, 1]
    noisy_img[rows, cols] = 0
    denoised_img = median_filter(noisy_img, size=5)
    axs[0, i].imshow(denoised_img, cmap='gray')
    axs[0, i].set_title(f'Uniform - {noise_ratio * 100}%')
    axs[0, i].axis('off')

    mean = np.random.uniform(0, 255)
    std = np.random.uniform(10, 50) * noise_ratio
    noise = np.random.normal(mean, std, img.shape)
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    denoised_img = median_filter(noisy_img, size=5)
    axs[1, i].imshow(denoised_img, cmap='gray')
    axs[1, i].set_title(f'Gaussian - {noise_ratio * 100}%')
    axs[1, i].axis('off')

    noise = np.random.exponential(scale=1, size=img.shape) * noise_ratio * 10
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    denoised_img = median_filter(noisy_img, size=5)
    axs[2, i].imshow(denoised_img, cmap='gray')
    axs[2, i].set_title(f'Exponential - {noise_ratio * 100}%')
    axs[2, i].axis('off')

plt.tight_layout()
plt.show()
print("finished part C")
