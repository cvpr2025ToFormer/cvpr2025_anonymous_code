import os
import random
import argparse
import json
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Custom json generator")

parser.add_argument('--path_root', type=str, required=False,
                    default='',
                    help="Path to Custom dataset")

parser.add_argument('--path_out', type=str, required=False,
                    default='../data_json', help="Output path")
parser.add_argument('--name_out', type=str, required=False,
                    default='custom_v3.json', help='Output file name')
parser.add_argument('--val_num', type=float, required=False,
                    default=750, help='Validation data num')
parser.add_argument('--test_num', type=float, required=False,
                    default=1500, help='Validation data num')
parser.add_argument('--seed', type=int, required=False, default=7240,
                    help='Random seed')

args = parser.parse_args()
random.seed(args.seed)

def check_dir_existence(path_dir):
    assert os.path.isdir(path_dir), \
        "Directory does not exist : {}".format(path_dir)

def check_file_existence(path_file):
    assert os.path.isfile(path_file), \
        "File does not exist : {}".format(path_file)

def main():
    check_dir_existence(args.path_root)
    check_dir_existence(args.path_out)

    folder_names = os.listdir(args.path_root)

    lines_all = []
    lines_val = []

    for folder in folder_names:
        if folder != "val_test_seq":
            subpath = os.path.join(args.path_root, folder, "color")
            color_names = os.listdir(subpath)

            for color_name in color_names:
                lines_all.append(folder + "/color/{}".format(color_name))
        else:
            val_test_path = os.path.join(args.path_root, "val_test_seq")
            val_test_folder_names = os.listdir(val_test_path)

            for val_folder in val_test_folder_names:
                val_subpath = os.path.join(val_test_path, val_folder, "color")
                val_color_names = os.listdir(val_subpath)

                for val_color_name in val_color_names:
                    lines_val.append("val_test_seq/" + val_folder + "/color/{}".format(val_color_name))

    all_num = len(lines_all) + len(lines_val)
    val_num = args.val_num
    test_num = args.test_num
    train_num = all_num - val_num - test_num
    print("Dataset Size: {}".format(all_num))
    print("Seen Size: {}".format(len(lines_all)))
    print("Uneen Size: {}".format(len(lines_val)))
    print("Trainset Size: {}".format(train_num))
    print("Valset Size: {}".format(val_num))
    print("Testset Size: {}".format(test_num))

    random.shuffle(lines_all)
    random.shuffle(lines_val)

    list_train = []
    list_val_test = []
    list_val = []
    list_test = []
    dict_json = {}
    cnt = 0
    for line in lines_all:
        cnt = cnt + 1
        pack = line.split("/")[0]
        time_stamp = line.split("/")[-1]
        time_stamp = time_stamp.split(".")[0]
        gt_name = pack + "/gt/" + time_stamp + ".png"
        guide_name = pack + "/sparse/" + time_stamp + ".png"
        dict_sample = {'rgb_name': line, 'gt_name': gt_name, 'guide_name': guide_name}
        if cnt <= train_num:
            list_train.append(dict_sample)
        else:
            list_val_test.append(dict_sample)

    for line in lines_val:
        pack = line.split("/")[1]
        time_stamp = line.split("/")[-1]
        time_stamp = time_stamp.split(".")[0]
        gt_name = "val_test_seq/" + pack + "/gt/" + time_stamp + ".png"
        guide_name = "val_test_seq/" + pack + "/sparse/" + time_stamp + ".png"
        dict_sample = {'rgb_name': line, 'gt_name': gt_name, 'guide_name': guide_name}
        list_val_test.append(dict_sample)

    random.shuffle(list_val_test)
    list_val = list_val_test[0:val_num]
    list_test = list_val_test[val_num:val_num+test_num]

    dict_json['train'] = list_train
    dict_json['val'] = list_val
    dict_json['test'] = list_test

    # Write to json files
    f = open(args.path_out + '/' + args.name_out, 'w')
    json.dump(dict_json, f, indent=4)
    f.close()

    print("Json file generation finished.")

if __name__ == '__main__':
    print('\nArguments :')
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))
    print('')

    main()
