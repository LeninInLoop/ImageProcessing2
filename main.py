import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


def load_image_as_array(image_path: str) -> np.ndarray:
    try:
        image = Image.open(image_path)
        image_array = np.array(image)
    except Exception as e:
        error_msg = f""" ↓\n
{'*' * 50 + "\n"}
Error loading image: {e}
{"\n"+'*' * 50}
""".strip()
        raise Exception(error_msg)
    return image_array


def plot_images_by_size(
        original_image,
        upsampled_images,
        save_path=None,
        titles=None,
) -> None:
    images = [original_image] + upsampled_images
    n_images = len(images)

    width_ratios = [img.shape[1] for img in images]

    if titles is None:
        titles = ["Original"] + [f"Upsampled {i+1}" for i in range(n_images-1)]

    scale = 0.01
    total_width = sum(width_ratios) * scale
    # Use the maximum height among the images to set the figure height.
    fig_height = max(img.shape[0] for img in images) * scale

    fig = plt.figure(figsize=(total_width, fig_height))
    gs = gridspec.GridSpec(1, n_images, width_ratios=width_ratios)

    # Plot each image in its own subplot.
    for idx, img in enumerate(images):
        ax = fig.add_subplot(gs[idx])
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(titles[idx])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def nearest_neighbor_upsample(original_img, new_height=400, new_width=400):
    orig_height, orig_width = original_img.shape

    rows_new = np.arange(new_height)
    cols_new = np.arange(new_width)

    src_rows = np.round(rows_new * (orig_height / new_height)).astype(int)
    src_cols = np.round(cols_new * (orig_width / new_width)).astype(int)

    # Perform advanced indexing to create the upsampled image
    upsampled_img = original_img[src_rows[:,None], src_cols]

    return upsampled_img

def bilinear_upsample(original_img, new_height=400, new_width=400):
    orig_height, orig_width = original_img.shape

    rows_new = np.arange(new_height); scale_factor_x = orig_width / new_width
    cols_new = np.arange(new_width); scale_factor_y = orig_height / new_height

    x = rows_new * scale_factor_x; x1 = np.floor(x).astype(int); x2 = np.minimum(x1 + 1, orig_width - 1)
    y = cols_new * scale_factor_y; y1 = np.floor(y).astype(int); y2 = np.minimum(y1 + 1, orig_height - 1)

    a = x - x1; b = y - y1

    # Perform advanced indexing to create the upsampled image
    upsampled_img = (1-a)[:,None]*(1-b)*original_img[x1[:,None], y1] + \
                    a[:,None]*(1-b)*original_img[x2[:,None], y1] + \
                    (1-a)[:,None]*b*original_img[x1[:,None], y2] + \
                    a[:,None]*b*original_img[x2[:,None], y2]

    return upsampled_img

def main():
    image_path = r"C:\Users\ADIB\Desktop\Image Processing\2\Adib_Nikjou_403114114_DIP_2\Images\cameraman.bmp"
    image_array = load_image_as_array(image_path)

    print(50*"-", "\nImage Array:\n", image_array)
    upsampled_image_nn = nearest_neighbor_upsample(image_array)
    print(50*"-", "\nUpsampled Image Array(N.N):\n", upsampled_image_nn)

    upsampled_image_bl = bilinear_upsample(image_array)
    print(50 * "-", "\nUpsampled Image Array(Bilinear):\n", upsampled_image_bl)

    plot_images_by_size(
        original_image=image_array,
        upsampled_images=[upsampled_image_nn, upsampled_image_bl],
        save_path = "Upsampled_images.png",
        titles = ["Original Image", "Upsampled Image(NN)", "Upsampled Image(Bilinear)"],
    )
    print(50 * "-", "\nUpsampled and Original images are saved as: ", "./Upsampled_images.png")

if __name__ == '__main__':
    main()