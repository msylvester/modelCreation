import requests
import time

def get_all_sets():
    """
    Retrieves all LEGO sets from the Rebrickable API and stores their IDs in an array.

    Returns:
        list: A list containing all the set IDs retrieved.
    """
    url = "https://rebrickable.com/api/v3/lego/sets/?page=1&page_size=100"
    headers = {
        'Accept': 'application/json',
        'Authorization': 'key fa4e68769dd0b1fe04ec4055eb870304'
    }
    sets_id = []

    while url:
        try:
            response = requests.get(url, headers=headers, timeout=10)  # 10-second timeout
            if response.status_code == 429:  # Handle rate limit exceeded
                print("Rate limit exceeded. Waiting before retrying...")
                time.sleep(60)  # Wait for a minute before retrying
                continue

            response.raise_for_status()  # Check for HTTP errors
            
            data = response.json()
            sets = data['results']
            
            # Store set IDs in the array and print them
            for lego_set in sets:
                set_id = lego_set.get('set_num', 'No set number available')
                sets_id.append(set_id)
                print(set_id)

            # Get the next page URL from the response
            url = data.get('next')
            if url is not None:
                print(f"Fetching next page: {url}")
            else:
                print("No more pages to fetch.")
                break
            
            time.sleep(1)  # Optional: Add a small delay between requests

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            break

    return sets_id

def main():
    sets_id = get_all_sets()
    print(f"Total sets retrieved: {len(sets_id)}")
    
    # Write all set IDs to a text file
    with open('sets_id.txt', 'w') as file:
        for set_id in sets_id:
            file.write(set_id + '\n')
    print("Set IDs have been written to 'sets_id.txt'")

if __name__ == "__main__":
    main()
