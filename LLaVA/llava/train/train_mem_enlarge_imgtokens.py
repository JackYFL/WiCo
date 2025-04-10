from llava.train.train_enlarge_imgtokens import train_enlarge_imgtokens

if __name__ == '__main__':
    train_enlarge_imgtokens(attn_implementation="flash_attention_2")
