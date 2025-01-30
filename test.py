import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig,BrowserConfig
from termcolor import colored

async def main():
    # Single JS command
    config = CrawlerRunConfig(
        js_code="window.scrollTo(0, document.body.scrollHeight);"
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://news.ycombinator.com",  # Example site
            config=config
        )
        print(colored(f"Crawled length: {len(result.links)}", "yellow"))

    # Multiple commands
    js_commands = [
        "window.scrollTo(0, document.body.scrollHeight);",
        # 'More' link on Hacker News
        "document.querySelector('a.morelink')?.click();",  
    ]
    config = CrawlerRunConfig(scan_full_page=True,screenshot=True,scroll_delay=10)
    browser_cfg = BrowserConfig(
    browser_type="chromium",
    headless=False,
    )
    async with AsyncWebCrawler(config = browser_cfg) as crawler:
        result = await crawler.arun(
            url="https://news.ycombinator.com",  # Another pass
            config=config
        )
        print(colored(f"After scroll+click, length: {len(result.links)}","green"))

if __name__ == "__main__":
    asyncio.run(main())
