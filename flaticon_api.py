from requests import Session, Request
import flaticon_api_key


# A Class to easily call the Flaticon Api
class FlaticonApi:

    # Constructor
    def __init__(self):
        self.api_url = 'https://api.flaticon.com/v3'
        self.api_key = flaticon_api_key.key
        self.headers = {'Accept': 'application/json', 'Authorization': 'string'}
        self.session = Session()
        self.session.headers.update(self.headers)
        self._get_token()

    # Obtain a token to use the API
    def _get_token(self):
        url = self.api_url + '/app/authentication'
        post = Request('POST', url, headers=self.headers, data={'apikey': self.api_key})
        prepared_post = post.prepare()

        resp = self.session.send(prepared_post)
        self.token = resp.json()['data']['token']
        self.headers['Authorization'] = "Bearer " + self.token
        self.session.headers.update(self.headers)

    # Get tags list
    def get_tags(self, page=1, limit=50):
        url = self.api_url + '/tags'
        if page <= 0:
            page = 1
        if limit < 50:
            limit = 50
        url += f'?page={page}&limit={limit}'
        resp = self.session.get(url)
        return resp.json()['data']

    # Get all Flaticon tags
    def get_all_tags(self):
        url = self.api_url + '/tags'
        resp = self.session.get(url)
        return resp.json()['data']

    def get_black_icons(self, search_term="", limit=100):
        resp_metadata = {}
        resp_data = []

        # Create the URL
        url = self.api_url + '/search/icons/priority?styleColor=black'
        if search_term != "":
            url += f'&q={search_term}'

        # Do a first search to get metadata
        resp = self.session.get(url)
        if resp.status_code == 200:
            resp_metadata = resp.json()['metadata']
        else:
            print("Error: no result. Status code ", resp.status_code)
            return resp_data, resp_metadata

        # While limit is not reached and there's still results, retrieve icons
        total_icons = resp_metadata['total']
        retrieved_ic_count = 0
        page = 1
        while retrieved_ic_count < total_icons and retrieved_ic_count < limit:
            resp = self.session.get(url + f'&page={page}')
            if resp.status_code == 200:
                resp_data = [*resp_data, *resp.json()['data']]
                retrieved_ic_count += 100
                page += 1
            else:
                print("Error: no result. Status code ", resp.status_code)
                return resp_data, resp_metadata
        return resp_data, resp_metadata



