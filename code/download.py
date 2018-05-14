# -*- coding:utf-8 -*-
import requests
from bs4 import BeautifulSoup
import json, re, os
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from lxml import etree
chromedriver = r"D:\software\chrome\chrome_driver\chromedriver"
os.environ["webdriver.chrome.driver"] = chromedriver



def getData(url, links):
    # driver = webdriver.PhantomJS(executable_path=r"C:\software\PhantomJS\bin\phantomjs.exe")#使用下载好的phantomjs，网上也有人用firefox，chrome，但是我没有成功，用这个也挺方便
    # driver.set_page_load_timeout(30)
    driver = webdriver.Chrome(chromedriver)  # 模拟打开浏览器
    driver.get("http://ibaotu.com/?m=login&a=snsLogin&type=weixin")  # 打开网址
    # driver.get('https://ibaotu.com/ppt/3-0-0-0-0-1.html')
    # assert "Python" in driver.title
    # elem = driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div[1]/nav/ul/li[6]/a[2]')
    # elem.send_keys("selenium")
    # elem.send_keys(Keys.RETURN)
    # assert "Google" in driver.title
    # driver.close()
    # driver.quit()
    # time.sleep(3)
    # html = driver.get(url[0])#使用get方法请求url，因为是模拟浏览器，所以不需要headers信息
    for page in range(452):
        html = driver.page_source#获取网页的html数据
        soup = BeautifulSoup(html, 'lxml')#对html进行解析，如果提示lxml未安装，直接pip install lxml即可
        table = soup.find_all('a', attrs={"class": "down-btn gradient-hor-og"})
        num = len(table)
        for i in range(num):
            link = 'http:' + table[i].get('href')
            htm = driver.get(link)
            htm = driver.page_source  # 获取网页的html数据
            sou = BeautifulSoup(html, 'lxml')  # 对html进行解析，如果提示lxml未安装，直接pip install lxml即可
            tabl = soup.find_all('a', attrs={"id": "downvip"})
            links.append(link)
        driver.find_element_by_link_text(u"下一页").click()#利用find_element_by_link_text方法得到下一页所在的位置并点击，点击后页面会自动更新，只需要重新获取driver.page_source即可。

def jsonDump(_json,name):
    """store json data"""
    with open(curpath+'/'+name+'.json','a') as outfile:
        json.dump(_json,outfile,ensure_ascii=False)
    with open(curpath+'/'+name+'.json','a') as outfile:
        outfile.write(',\n')

if __name__ == '__main__':
    url=['https://ibaotu.com/ppt/3-0-0-0-0-1.html'] #yzc为文件名，此处输入中文会报错，前面加u也不行，只好保存后手动改文件名……
    links = []
    getData(url, links)#调用函数