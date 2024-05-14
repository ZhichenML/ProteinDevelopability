import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler('sab_tap.log'), logging.StreamHandler()])

def submit_hlchain_to_tap(heavy_chain_sequence, light_chain_sequence):
    url = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/tap"
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Referer': 'https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/tap',
        'Origin': 'https://opig.stats.ox.ac.uk',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    }

    data = {"hchain": heavy_chain_sequence, "lchain": light_chain_sequence}

    with requests.Session() as session:
        try:
            response = session.post(url, headers=headers, data=data)
            response.raise_for_status()
            logging.info("Sequence data submitted successfully!")

            time.sleep(50)  # Wait for the server to process the request

            response = session.get(response.url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            log_file_box = soup.find('div', class_='log_file_box')
            if log_file_box:
                log_text = log_file_box.get_text(strip=True)
                ind = log_text.find('Summary')
                res = log_text[ind:] if ind != -1 else "Summary not found"
            else:
                logging.warning('Could not find the specified div tag.')
                res = "Log file box not found"

        except requests.RequestException as e:
            logging.error(f"Request failed: {e}")
            res = "Request failed"
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            res = "An error occurred"

    res = ' '.join(res.split())
    res = res[res.find('Total IMGT CDR Length'):] if 'Total IMGT CDR Length' in res else res
    res = res.replace('(GREEN flag)', '\n')
    res_list = [v.split(':') for v in res.split('\n')[:-1]]
    res_stripped = [[element.strip(' score is') for element in sublist] for sublist in res_list]

    # Extract column names from the first row of res_stripped
    columns = [col[0] for col in res_stripped]

    # Extract data rows
    data_rows = [col[1] for col in res_stripped]

    # Create DataFrame
    res_df = pd.DataFrame([data_rows], columns=columns)

    # Add heavy and light sequences as new columns
    res_df['Heavy Sequence'] = heavy_chain_sequence
    res_df['Light Sequence'] = light_chain_sequence

    # Reorder columns to move 'Heavy Sequence' and 'Light Sequence' to the left
    columns_order = ['Heavy Sequence', 'Light Sequence'] + [col for col in res_df.columns if col not in ['Heavy Sequence', 'Light Sequence']]
    res_df = res_df[columns_order]
    
    return res_df

def main():
    setup_logging()
    input_path = r'C:\Users\A\Desktop\sab_tap\TheraSAbDab_SeqStruc_OnlineDownload.csv'
    output_path = r'C:\Users\A\Desktop\sab_tap\TheraSAbDab_Results.csv'

    df = pd.read_csv(input_path)
    results = []

    for i in range(len(df)):
        heavy_chain_sequence = df.loc[i, 'Heavy Sequence']
        light_chain_sequence = df.loc[i, 'Light Sequence']
        logging.info(f"Processing sequence {i + 1}/{len(df)}")
        res_df = submit_hlchain_to_tap(heavy_chain_sequence, light_chain_sequence)
        
        
        results.append(res_df)
        time.sleep(5)  # Wait before sending the next request
    
    result_df = pd.concat(results, ignore_index=True)
    result_df.to_csv(output_path, index=False)
    logging.info("Results saved to CSV")

if __name__ == '__main__':
    main()
