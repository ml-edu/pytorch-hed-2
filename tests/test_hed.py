import torchHED
import PIL


def test_hed_on_image():
    img = PIL.Image.open("./images/sample.png")
    torchHED.process_img(img)


def test_hed_on_file():
    torchHED.process_file("./images/sample.png", "./images/torchHED.png")


def test_hed_on_folder():
    torchHED.process_folder("./tests/test_imgs", "./tests/tests_imgs_out")
