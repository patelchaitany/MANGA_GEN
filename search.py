from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from langchain_community.tools import DuckDuckGoSearchResults
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
import asyncio
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
import os

# Adjust the number of results returned by DuckDuckGo
duckduckgo_search = DuckDuckGoSearchResults(output_format="list", num_results=10)  # Example: increase to 50 results

crawl4ai_search = AsyncWebCrawler()

async def perform_search(query=None, search_engine="duckduckgo"):
    """
    Perform a search, crawl URLs, and return website content.
    """
    if query is None:
        query = input("Enter your search query: ")
    urls = []
    if search_engine == "duckduckgo":
        print("Performing search with DuckDuckGo...")
        search_results = duckduckgo_search.run(query)
        urls = search_results  # DuckDuckGoSearchResults returns a list
    else:
        raise ValueError(f"Invalid search engine: {search_engine}. Choose 'crawl4ai' or 'duckduckgo'.")

    print(f"URLs fetched from {search_engine}: {urls}")

    crawled_contents = []
    browser_config = BrowserConfig(headless=True, verbose=False)

    bm25_filter = BM25ContentFilter(
        user_query=query,
        # Adjust for stricter or looser results
        bm25_threshold=1.2
    )
    md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=10
    )
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        stream=False,  # Default: get all results at once
        markdown_generator=md_generator,
        scan_full_page=True,
        scroll_delay=0.2,
        delay_before_return_html=1,
        simulate_user=True,
        wait_for_images=True,
        wait_until="domcontentloaded",
        exclude_external_images=True
    )
    crawled_contents = []
    async with AsyncWebCrawler(config=browser_config) as crawler:  # Create crawler instance here
        final_url = [url['link'] for url in urls]
        results = await crawler.arun_many(
            urls=final_url,
            config=run_config,
            dispatcher=dispatcher
        )
        for result in results:
            if result.success:
                filtered_images = []
                bm25_filter_image_desc = BM25ContentFilter(
                    user_query=query,
                    bm25_threshold=0.2
                )
                for i in result.media.get("images", []):
                    scr = i.get('src', None)
                    url = result.url
                    alt = i.get('desc', '')
                    score = i.get('score', 0)
                    if scr and scr.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp')) and score>=6:
                        filtered_images.append({
                            'score': score,
                            'link': scr,
                            'src+url': url + scr,
                        })
                crawled_contents.extend(filtered_images)
            else:
                print(f"failed")

    # Filter and sort images by score (only include scores greater than 6)
    sorted_images = sorted(
        [img for img in crawled_contents if float(img.get('score', 0)) >= 6],
        key=lambda x: float(x.get('score', 0)),
        reverse=True
    )

    # Prepare table headers
    table_header = "| Score | Link | Src+Url |\n|---|---|---|\n"
    table_rows = ""
    for image in sorted_images:
        table_rows += f"| {image.get('score', '')} | {image.get('link', '')} | {image.get('src+url', '')} |\n"

    # print(table_header + table_rows)
    return sorted_images

async def crawl_website(url):
    """
    Crawl a website using Crawl4AI and return the text content.

    Args:
        url (str): The URL of the website to crawl.
        num_links_to_parse (int, optional): The maximum number of links to parse on the website. Defaults to None (unlimited).

    Returns:
        list: A list of text content from the crawled pages.
    """
    browser_config = BrowserConfig(
        headless=False,
        use_managed_browser=True,
    )
    crawler_config = CrawlerRunConfig(
        stream=False,  # Delay between scroll steps (in seconds)
        scan_full_page=True,
        scroll_delay=0.2,
        delay_before_return_html=1,
        cache_mode=CacheMode.BYPASS,
        simulate_user=True,
        wait_until="domcontentloaded",
    )
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=90.0,
        check_interval=1.0,
        max_session_permit=10
    )
    print(f"Crawling website: {url} with a limit of links to parse...")
    crawled_contents = []
    async with AsyncWebCrawler(config=browser_config) as crawler:  # Create crawler instance here
        final_url = [url, "https://example.com/"]
        results = await crawler.arun_many(
            urls=final_url,
            config=crawler_config,
            dispatcher=dispatcher
        )

        for result in results:
            if result.success:
                crawled_contents.append(result.media.get("images", []))
            else:
                print(f"failed")

    text_contents = [results[0].media]  # Assuming results contain a 'text' key
    return crawled_contents

def main():
    while True:
        print("\nChoose an option:")
        print("1. Perform a search")
        print("2. Crawl a website")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")

        if choice == "1":
            query = input("Enter your search query: ")
            urls = asyncio.run(perform_search(query=query))
        elif choice == "2":
            url = input("Enter the URL to crawl: ")
            try:
                text_content = asyncio.run(crawl_website(url))
            except Exception as e:
                print(f"Error: {e}")
        elif choice == "3":
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
