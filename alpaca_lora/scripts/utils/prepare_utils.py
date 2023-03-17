import json
import argparse


def split_json(alpaca_data):

    json_data = json.load(open(alpaca_data))
    print("load aplaca data number = {}".format(len(json_data)))
    train_src = alpaca_data.replace("alpaca_data.json", 'train.src')
    train_tgt = alpaca_data.replace("alpaca_data.json", 'train.tgt')

    with open(train_src, 'w') as f_src, open(train_tgt, 'w') as f_tgt:
        for data in json_data:
            src = data['instruction']
            if len(data['input']) > 0:
                src += " " + data['input']
            tgt = data['output']
            f_src.writelines(src.replace("\n", '<0x0A>') + '\n')
            f_tgt.writelines(tgt.replace("\n", '<0x0A>') + '\n')

def replace_data(alpaca_data):

    train_src = alpaca_data.replace("alpaca_data.json", 'train.spm.src.tmp')
    train_tgt = alpaca_data.replace("alpaca_data.json", 'train.spm.tgt.tmp')

    valid_src = alpaca_data.replace("alpaca_data.json", 'valid.spm.src.tmp')
    valid_tgt = alpaca_data.replace("alpaca_data.json", 'valid.spm.tgt.tmp')

    train_files = [train_src, train_tgt, valid_src, valid_tgt]
    for train_file_new in train_files:
        
        train_src_rep = train_file_new.replace(".tmp", "")
        with open(train_src_rep, 'w') as f_o:
            for line in open(train_file_new).readlines():
                newline = line.replace("▁< 0 x 0 A >", " <0x0A> ").replace("< 0 x 0 A >", " <0x0A> ")
                newline = newline.replace("  ", " ")
                f_o.writelines(newline)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manner",
        required=True,
        type=str,
        default="split",
        help="process utils",
    )
    parser.add_argument(
        "--alpaca-data",
        default="/opt/data/private/data/llama_new/alpaca_data.json",
        help="where in model_dir are weights saved",
    )    
    
    args = parser.parse_args()
    print(args.alpaca_data)
    if args.manner == "split":
        split_json(args.alpaca_data)
    elif args.manner == "replace":
        replace_data(args.alpaca_data)
    else:
        print("No Support!")


if __name__ == "__main__":
    main()
