from icrawler.builtin import BingImageCrawler

keyWord = input('검색어 : ')
crawlNum = int(input('원하는 이미지 수 : '))

bc = BingImageCrawler(feeder_threads = 1,
                      parser_threads = 1,
                      downloader_threads = 4,
                      storage = {'root_dir':'img//'+keyWord}
                      )
bc.crawl(keyword=keyWord, filters=None, offset=0, max_num=crawlNum)