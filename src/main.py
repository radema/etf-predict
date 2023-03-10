from utils import *



def main():
    data = ExtractData(TICKERS)
    data.to_parquet('data/tickers_source.parquet')
    print('\nData stored!')

    
    

if __name__ == '__main__':
    main()