import torch.nn as nn


class DCGenerator(nn.Module):
    def __init__(self, z_dim=20, image_size=64):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, image_size * 8, kernel_size=4, stride=1),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(image_size * 8, image_size * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(image_size * 4, image_size * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size * 2),
            nn.ReLU(inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(image_size * 2, image_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True),
        )

        self.last = nn.Sequential(
            nn.ConvTranspose2d(image_size, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        output = self.layer1(z)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.last(output)

        return output


class DCDiscriminator(nn.Module):
    def __init__(self, z_dim=20, image_size=64):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, image_size, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(image_size, image_size*2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(image_size*2, image_size*4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(image_size*4, image_size*8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.last = nn.Conv2d(image_size*8, 1, kernel_size=4, stride=1)

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.last(output)

        return output


def gd_test():
    import matplotlib.pyplot as plt
    import torch

    G = DCGenerator(z_dim=20, image_size=64)
    D = DCDiscriminator(z_dim=20, image_size=64)

    input_z = torch.randn(1, 20)
    input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)

    fake_images = G(input_z)
    d_out = D(fake_images)
    print(nn.Sigmoid()(d_out))

    img_transformed = fake_images[0][0].detach().numpy()
    plt.imshow(img_transformed, 'gray')
    plt.show()


if __name__ == "__main__":
    gd_test()
