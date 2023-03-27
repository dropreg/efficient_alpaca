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

def extract_translation_data(trans_data):
    total_number = 50000
    zh_prefix = "翻译成英文"
    en_prefix = "translate to Chinese"
    with open(trans_data.replace("translation2019zh_train.json", 'train.src'), 'w') as f_src, \
        open(trans_data.replace("translation2019zh_train.json", 'train.tgt'), 'w') as f_tgt:
        for line_idx, lines in enumerate(open(trans_data).readlines()):
            if line_idx > total_number:
                break
            json_data = json.loads(lines)
            en_input = en_prefix + ": " + json_data["english"]
            zh_input = zh_prefix + ": " + json_data["chinese"]
            en_output = json_data["chinese"]
            zh_output = json_data["english"]

            f_src.writelines(en_input + '\n')
            f_tgt.writelines(en_output + '\n')

            f_src.writelines(zh_input + '\n')
            f_tgt.writelines(zh_output + '\n')

def split_zh_json(alpaca_data):

    print("load aplaca data {}".format(alpaca_data))
    train_src = alpaca_data.replace("Belle.train.json", 'train.src')
    train_tgt = alpaca_data.replace("Belle.train.json", 'train.tgt')

    with open(train_src, 'w') as f_src, open(train_tgt, 'w') as f_tgt:
        for lines in open(alpaca_data).readlines():
            data = json.loads(lines)
            src = data['input'].strip()
            tgt = data['target'].strip()
            if len(src) > 0 and len(tgt) > 0:
                f_src.writelines(src.replace("\n", '<0x0A>').replace("\\n", '<0x0A>').replace("\r\n", '<0x0A>').replace("\r", '<0x0A>') + '\n')
                f_tgt.writelines(tgt.replace("\n", '<0x0A>').replace("\\n", '<0x0A>').replace("\r\n", '<0x0A>').replace("\r", '<0x0A>') + '\n')

def replace_zh_data(alpaca_data):

    train_src = alpaca_data.replace("Belle.train.json", 'train.spm.src.tmp')
    train_tgt = alpaca_data.replace("Belle.train.json", 'train.spm.tgt.tmp')

    valid_src = alpaca_data.replace("Belle.train.json", 'valid.spm.src.tmp')
    valid_tgt = alpaca_data.replace("Belle.train.json", 'valid.spm.tgt.tmp')

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
        help="alpaca self-instruction data_dir",
    )
    parser.add_argument(
        "--translation-data",
        default="/opt/data/private/data/llama/trans/translation2019zh_train.json",
        help="transltion data_dir",
    )
    args = parser.parse_args()

    # print(args.alpaca_data, args.translation_data)
    if args.manner == "split":
        split_json(args.alpaca_data)
    elif args.manner == "replace":
        replace_data(args.alpaca_data)
    elif args.manner == "trans":
        extract_translation_data(args.translation_data)
    elif args.manner == "split_zh":
        split_zh_json(args.alpaca_data)
    elif args.manner == "replace_zh":
        replace_zh_data(args.alpaca_data)
    else:
        print("No Support!")


if __name__ == "__main__":
    main()
