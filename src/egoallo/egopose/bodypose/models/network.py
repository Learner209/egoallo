from torch import nn


nn.Module.dump_patches = True


class EgoExo4D(nn.Module):
    def __init__(self, input_dim, output_dim, num_layer, embed_dim, nhead, device, opt):
        super(EgoExo4D, self).__init__()

        self.linear_embedding = nn.Linear(input_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layer,
        )

        self.stabilizer = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )
        self.joint_rotation_decoder = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 126),
        )

    def forward(self, input_tensor, image=None, do_fk=True):
        input_tensor = input_tensor.reshape(
            input_tensor.shape[0],
            input_tensor.shape[1],
            -1,
        )  # BS x T x 3
        x = self.linear_embedding(input_tensor)  # BS x T x D
        x = x.permute(1, 0, 2)  # T x BS x D
        x = self.transformer_encoder(x)  # T x BS x D
        x = x.permute(1, 0, 2)[:, -1]  # BS x D
        global_orientation = self.stabilizer(x)  # BS x D_o
        return global_orientation
