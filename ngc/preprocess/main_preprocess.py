import argparse
from avatar import Avatar 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help="Name of the object")
    parser.add_argument('--mesh_path', type=str, required=True, help="path of mesh file")
    parser.add_argument('--skel_file', type=str, required=True, help="Skeletal file information")
    parser.add_argument('--corres_file', type=str, required=True, help="Correspondence file information")
    #parser.add_argument('--output_folder', type=str, required=True, help="Output folder information")

    args = parser.parse_args()

    avatar = Avatar(args)


