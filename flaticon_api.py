from requests import Session, Request
import api_key


# Build up a class so we can easily make the REST API calls
class REACH:
    def __init__(self, api_key):
        self.token = None
        self.apiurl = 'https://api.flaticon.com/v3'
        self.api_key = api_key
        self.headers = {'Accept': 'application/json', 'Authorization': 'string'}
        self.session = Session()
        self.session.headers.update(self.headers)

    def get_token(self):
        url = self.apiurl + '/app/authentication'
        post = Request('POST', url, headers=self.headers, data={'apikey': self.api_key})
        prepared_post = post.prepare()

        resp = self.session.send(prepared_post)
        self.token = resp.json()['data']['token']
        self.headers['Authorization'] = "Bearer " + self.token
        self.session.headers.update(self.headers)

    def get_tags(self):
        url = self.apiurl + '/tags'
        resp = self.session.get(url)
        print(resp.json())

    def get_black_icons(self):
        url = self.apiurl + '/search/icons/priority?styleColor=black'
        resp = self.session.get(url)
        print(resp.status_code)
        data = resp.json()['data']
        print(data)
        return data


reach = REACH(api_key.apiKey)
reach.get_token()
# reach.get_tags()
reach.get_black_icons()

