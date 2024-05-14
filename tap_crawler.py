import requests
import json

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from fake_useragent import UserAgent
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd


def submit_hlchain_to_tap(heavy_chain_sequence = "QVKLQESGAELARPGASVKLSCKASGYTFTNYWMQWVKQRPGQGLDWIGAIYPGDGNTRYTHKFKGKATLTADKSSSTAYMQLSSLASEDSGVYYCARGEGNYAWFAYWGQGTTVTVSS", 
                    light_chain_sequence = "DIELTQSPASLSASVGETVTITCQASENIYSYLAWHQQKQGKSPQLLVYNAKTLAGGVSSRFSGSGSGTHFSLKIKSLQPEDFGIYYCQHHYGILPTFGGGTKLEIK"):

    # 目标URL
    url = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/tap"

    header = {'authority':'opig.stats.ox.ac.uk',
    'method': 'POST',
    'path': '/webapps/sabdab-sabpred/sabpred/tap',
    'scheme': 'https',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Content-Type':'application/x-www-form-urlencoded',
    'Referer': 'https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/tap',
    'Origin':'https://opig.stats.ox.ac.uk',
    'Cookie':'_ga_R913GG5VNB=GS1.1.1715565431.1.0.1715565431.0.0.0; _ga=GA1.3.934786908.1715565431; _gid=GA1.3.448289092.1715565432; _gat_gtag_UA_142143190_1=1',
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    }

    # 构建要提交的数据
    data = {"hchain": heavy_chain_sequence, "lchain": light_chain_sequence}
  
    
    response = requests.post(url, headers=header, data=data) 
    response.raise_for_status()

    # 检查请求是否成功
    if response.status_code == 200:
        print("序列数据提交成功！")
    else:
        print("序列数据提交失败，状态码：", response.status_code)

    time.sleep(50)
    # 目标网址
    res_url = response.url
    
    # 发送HTTP请求
    response = requests.get(res_url)

    # 确保请求成功
    if response.status_code == 200:
        # 使用BeautifulSoup解析HTML内容
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 找到class为log_file_box的div标签
        log_file_box = soup.find('div', class_='log_file_box')
        
        # 检查是否找到了该标签
        if log_file_box:
            # 获取该标签内的文本内容
            log_text = log_file_box.get_text(strip=True)
            ind = log_text.find('Summary') 
            res = log_text[ind:]
            
        else:
            print('未能找到指定的div标签。')
    else:
        print('请求网页失败，状态码:', response.status_code)
    
    res = ' '.join(res.split())
    res = res[res.find('Total IMGT CDR Length'):]
    res = res.replace('(GREEN flag)', '\n')
    import pdb; pdb.set_trace()
    res =  [ [v.split(':')[0].strip(), v.split(':')[1].strip()]  for v in res.split('\n')[:-1] ]
    res = pd.DataFrame(res, columns=['TAP', 'value'])
    
    return res


def main():
    path = r'C:\Users\A\Desktop\sab_tap\TheraSAbDab_SeqStruc_OnlineDownload.csv'
    df = pd.read_csv(path)
    
    for i in range(len(df)):
        heavy_chain_sequence = df.loc[i, 'Heavy Sequence']
        light_chain_sequence = df.loc[i, 'Light Sequence']
        res = submit_hlchain_to_tap(heavy_chain_sequence, light_chain_sequence)
        import pdb; pdb.set_trace()
        print(res)


if __name__ == '__main__':
    main()
    



