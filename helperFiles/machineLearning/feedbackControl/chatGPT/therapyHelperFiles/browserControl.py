# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Navigate webpages
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium import webdriver

# -------------------------------------------------------------------------- #
# ---------------------------- ChatGPT Interface --------------------------- #
        
class browserControl:
    def __init__(self, initDummyPage = "www.google.com"):
        # Open a Chrome Browser (in Terminal) to Run Code
        # self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        self.driver = webdriver.Chrome()
        # self.open_url(initDummyPage)
        
    def open_url(self, url):
        # Open a URL in the same tab
        self.driver.get(url)
        
    # ---------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Instantiate class.
    browserController = browserControl()






