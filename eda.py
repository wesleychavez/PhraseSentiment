import pandas as pd

def main():
    train = pd.read_csv('train.tsv', sep="\t")
    test = pd.read_csv('test.tsv', sep="\t")
    print (train.columns)    
    print (test.columns)    
    print (train.dtypes)    

if __name__ == '__main__':
    main()
