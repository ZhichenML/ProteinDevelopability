import requests
import json

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from fake_useragent import UserAgent


# 序列数据，需要替换成你的实际数据
heavy_chain_sequence = "QVKLQESGAELARPGASVKLSCKASGYTFTNYWMQWVKQRPGQGLDWIGAIYPGDGNTRYTHKFKGKATLTADKSSSTAYMQLSSLASEDSGVYYCARGEGNYAWFAYWGQGTTVTVSS"
light_chain_sequence = "DIELTQSPASLSASVGETVTITCQASENIYSYLAWHQQKQGKSPQLLVYNAKTLAGGVSSRFSGSGSGTHFSLKIKSLQPEDFGIYYCQHHYGILPTFGGGTKLEIK"

# 目标URL
url = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/tap"

# 设置Selenium WebDriver
# driver = webdriver.Chrome(executable_path="C:\\Users\\A\\Downloads\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe")  # 或使用其他浏览器驱动
# driver = webdriver.Chrome(service=webdriver.chrome.service.Service(executable_path="C:\\Users\\A\\Downloads\\chromedriver-win64\\chromedriver-win64\\chromedriver"))
service = Service()
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)

# service = Service(ChromeDriverManager().install())
# driver = webdriver.Chrome(service=service)

header = {'authority':'opig.stats.ox.ac.uk',
'method': 'POST',
'path': '/webapps/sabdab-sabpred/sabpred/tap',
'scheme': 'https',
'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
'Accept-Encoding': 'gzip, deflate, br, zstd',

# 'Content-Length':'245',
'Content-Type':'application/x-www-form-urlencoded',

'Referer': 'https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/tap',
'Origin':'https://opig.stats.ox.ac.uk',
'Cookie':'_ga_R913GG5VNB=GS1.1.1715565431.1.0.1715565431.0.0.0; _ga=GA1.3.934786908.1715565431; _gid=GA1.3.448289092.1715565432; _gat_gtag_UA_142143190_1=1',
'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
}
try:
    # 打开网页
    driver.get(url)
    # 等待页面加载完成
    # time.sleep(5)  # 如果需要可以取消注释并设置合适的等待时间
    
    # 构建要提交的数据
    data = {"hchain": heavy_chain_sequence, "lchain": light_chain_sequence}
        
    # ua = UserAgent()
    # header = {'User-Agent':ua.random}
    # 发送POST请求，使用JSON格式
    
    # requests.request("POST", url, data=data)
    response = requests.post(url, headers=header, data=data) #, headers={'Content-Type': 'application/json'})
    response.raise_for_status()

    # 检查请求是否成功
    if response.status_code == 200:
        print("序列数据提交成功！")
    else:
        print("序列数据提交失败，状态码：", response.status_code)

    # 打印服务器响应的内容
    print(response.text)
    
    import requests
    from bs4 import BeautifulSoup

    # 目标网址
    url = response.url
    import pdb; pdb.set_trace()
    # 发送HTTP请求
    response = requests.get(url)

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

    import pdb; pdb.set_trace()
    # 查找<textarea>元素，这里假设<textarea>没有id属性，只有name属性
    # 如果有id属性，可以使用id更准确地查找元素
    textarea_element = driver.find_element(By.NAME, "lchain")

    # 清除原有的文本（如果有）
    textarea_element.clear()

    # 输入本地数据
    textarea_element.send_keys(light_chain_sequence)

    # 等待一段时间，确保数据已正确输入
    # time.sleep(5)  # 如果需要可以取消注释并设置合适的等待时间
    
    textarea_element = driver.find_element(By.NAME, "hchain")
    textarea_element.clear()
    textarea_element.send_keys(heavy_chain_sequence)

    submit_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
    
    submit_button.click()
    import pdb; pdb.set_trace()
finally:
    # 关闭浏览器
    driver.quit()



