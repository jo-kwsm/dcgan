import torch
import torch.nn as nn


class SAGenerator(nn.Module):
    def __init__(self, z_dim=20, image_size=64):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(z_dim, image_size * 8, kernel_size=4, stride=1)
            ),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(image_size * 8, image_size * 4, kernel_size=4, stride=2, padding=1)
            ),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(image_size * 4, image_size * 2, kernel_size=4, stride=2, padding=1)
            ),
            nn.BatchNorm2d(image_size * 2),
            nn.ReLU(inplace=True),
        )

        self.self_attention1 = Self_Attention(in_dim=image_size * 2)

        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(image_size * 2, image_size, kernel_size=4, stride=2, padding=1)
            ),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True),
        )

        self.self_attention2 = Self_Attention(in_dim=image_size)

        self.last = nn.Sequential(
            nn.ConvTranspose2d(image_size, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        output = self.layer1(z)
        output = self.layer2(output)
        output = self.layer3(output)
        output, attention_map1 = self.self_attention1(output)
        output = self.layer4(output)
        output, attention_map2 = self.self_attention2(output)
        output = self.last(output)

        return output, attention_map1, attention_map2


class SADiscriminator(nn.Module):
    def __init__(self, z_dim=20, image_size=64):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(1, image_size, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(image_size, image_size*2, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(image_size*2, image_size*4, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.self_attention1 = Self_Attention(in_dim=image_size*4)

        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(image_size*4, image_size*8, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.self_attention2 = Self_Attention(in_dim=image_size*8)

        self.last = nn.Conv2d(image_size*8, 1, kernel_size=4, stride=1)

    def forward(self, z):
        output = self.layer1(z)
        output = self.layer2(output)
        output = self.layer3(output)
        output, attention_map1 = self.self_attention1(output)
        output = self.layer4(output)
        output, attention_map2 = self.self_attention2(output)
        output = self.last(output)

        return output, attention_map1, attention_map2


class Self_Attention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.query_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim//8,
            kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim//8,
            kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=1
        )

        self.softmax = nn.Softmax(dim=-2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        X = x

        proj_query = self.query_conv(X).view(
            X.shape[0],
            -1,
            X.shape[2]*X.shape[3]
        )
        proj_query = proj_query.permute(0, 2, 1)
        proj_key = self.key_conv(X).view(
            X.shape[0],
            -1,
            X.shape[2]*X.shape[3]
        )

        S = torch.bmm(proj_query, proj_key)

        attention_map_T = self.softmax(S)
        attention_map = attention_map_T.permute(0, 2, 1)

        proj_value = self.value_conv(X).view(
            X.shape[0],
            -1,
            X.shape[2]*X.shape[3]
        )
        o = torch.bmm(
            proj_value,
            attention_map.permute(0, 2, 1)
        )

        o = o.view(X.shape[0], X.shape[1], X.shape[2], X.shape[3])
        out = x+self.gamma*o

        return out, attention_map



def gd_test():
    import matplotlib.pyplot as plt
    import torch

    G = SAGenerator(z_dim=20, image_size=64)
    D = SADiscriminator(z_dim=20, image_size=64)

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
