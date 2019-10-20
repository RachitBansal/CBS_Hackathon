!pip install -U sentence-transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
sent = ["I am Jeevesh Juneja . "]
embedding = model.encode(sent)
print(len(embedding))
import numpy as np
renew_sentences = ['I want to renew my policy . ', 'I want my policy renewed as soon as possible', 'How can i get my policy renewed?']
new_sentences = ['I want to get a new policy.',"I read about your new policy and now i can't wait to get it","I came across this new policy of yours about cars, can i get it?"]
check_sentences = ['How soon can i renew my policy?' , 'An accident happened , it was in the news, i was a victim there , can i get the new policy ?']
corpus = renew_sentences + new_sentences + check_sentences
corpus_embeddings = model.encode(corpus)
embeddings = []
for embedding in corpus_embeddings :
  embeddings.append(np.array(embedding))
print(np.dot(embeddings[6],embeddings[0])+np.dot(embeddings[6],embeddings[1])+np.dot(embeddings[6],embeddings[2]))
print(np.dot(embeddings[7],embeddings[0])+np.dot(embeddings[7],embeddings[1])+np.dot(embeddings[7],embeddings[2]))
print(np.dot(embeddings[6],embeddings[3])+np.dot(embeddings[6],embeddings[4])+np.dot(embeddings[6],embeddings[5]))
print(np.dot(embeddings[7],embeddings[3])+np.dot(embeddings[7],embeddings[4])+np.dot(embeddings[7],embeddings[5]))
import requests 

URL = '54.173.118.186/accounts/login'

data = { 'username' : ''
         'password' : ''
         'otp' : ''
         'business type' : ''}
r = requests.post( url=URL, data=data )

r = 
info_we_have = {}
def get_max_similarity(sentence, features) :
  sentence_embeddings = model.encode(sentence)
  feature_embeddings = model.encode(feature)
  lis = []
  for feature in feature_embeddings :
    lis.append(np.dot(sentence_embeddings[0],feature))
  i = lis.index(max(lis))
  lis = sort(lis)
  print(lis[0:5])
  return features[i]  #Or return list of features accoding to some threshold

def update_feature(s, val) :
  info_we_have[s] = val

def get_feature(s) :
  if s in info_we_have.keys() :
    return info_we_have[s]
  return -1

def kit_particular_details(sentence) :
  lis = get_max_similarity(sentence, kit_details)
  i = get_feature('kit_id') 
  kit = requests.get('23.22.237.185/zopperassure/kit/'+str(i))
  kit_details = kit.json()
  for feature in lis :
    print(feature)
    print(kit_details[feature])

def warranty_particular_details(sentence) :
  lis = get_max_similarity(sentence, warranty_details)
  i = get_feature('kit_id') 
  kit = requests.get('23.22.237.185/zopperassure/kit/'+str(i))
  kit_details = kit.json()
  for feature in lis :
    print(feature)
    print(kit_details[feature])
    
    def person_reply_data(req_str) :
  di = {}
  keys = req_str.split('\n')
  for key in keys :
    value = user_input()
    di += {key : value}
    

list1 = []
list2 = []
def recur(data) :
    if type(dic[data])!=dict :
      list1+=data
    else :
      list1.append(recur(list(dic[data].keys())))
    return list1
    
    
def create(task_str):         # task_str: either warranty creation or claim registration
  data_req = list(dic.get(task_str).keys())
  req_str = ""
  for data_r in data_req:
    list2 += recur(data_r)
    
  for elem in list2 :
    if elem not in info_we_have:
      req_str += str(elem) + "\n"
  chatbot_reply('Please enter the following details: ' + req_str)
  data = person_reply_data(req_str)
  info_we_have += data
  
  # --- post request to create warranty using prev_data[cust_id] ------
  
  
def chatbot_reply(tbw):
  print("Customer Care: ", tbw)
  return 

# def person_reply_data(req_str):
#   req_objs = req_str.split(" ")
  
  
  
  
def sentence_matching(query, sentences):
  query_emb = model.encode(query)
  sentences_emb = model.encode(sentences)
  i=0
  dot_prods = 0
  for sent in sentences_emb:
    dot_prods += np.dot(query_emb, sent)
    i += 1
  return dot_prods/i

ou = []     #cosine similarities between user input and sentences of top layer
keyword = []
def iterative(user_input) :
  for i in range(len(sentences)) :
    ou.append(sentence_matching(user_input, sentences[i]))
    keyword.append(sentences2[i])
  key_no = ou.index(max(ou))
  return key_no
tree = {'0' : keyword}
all_dot_prods = {'0' : ou}
def recurse(dictionary, user_input_embedding, layer_no=1) :
  lis = []
  keyword_lis=[]
  for key in dictionary.keys() :
    lis.append(np.dot(model.encode([key]), np.transpose(user_input_embedding)))
    keyword_lis.append(key)
  global all_dot_prods
  all_dot_prods[str(layer_no)] = lis
  tree[str(layer_no)] = keyword_lis
  for key in dictionary.keys() :
    if type(dictionary[key]) == dict :
      recurse(dictionary[key],user_input_embedding, layer_no+1)      
            



query = [input("User: ")]
query_emb = model.encode(query)
mp_top_index = iterative(query)
recurse(dic, query_emb)
print(mp_top_index)

perform_action(mp_top_index)

import matplotlib.pyplot as plt 
import numpy as np 

# 'coss' dictionary contains list of cosine similarities of each layer 

def analysis(coss):
  stats = []

  for layer in list[coss.keys()]:
    layer_coss = np.array(coss[layer])
    plt.plot(layer_coss)
    stats.append(list(np.mean(layer_coss), np.std(layer_coss), np.var(layer_coss)))
    print()
  plt.show()
warranty_listing = ['what all warranties are available', 'list all warranties', 'warranties list', 'I want all warranties', 'all warranties']
warranty_details = ['tell me my warranty details', 'give me my warranty details', 'warranty details', 'I want my warranty info', 'information regarding my warranty']
kit_details = ['give me my warranty kit details', 'information regarding my warranty kit', 'warranty kit', 'my warranty kit']
register_claim = ['please register claim', 'claim registration', 'register claim', 'I want to register a claim', 'registering new claim']
media_upload = ['upload media', 'upload a file', 'file uploading', 'please upload my file', 'submit a media file']
media_download = ['I want to get my file', 'download my file', 'file downloading', 'please download a file for me', 'extract media file']
warranty_creation = ['create a new warranty', 'i want to create new warranty', 'new warranty', 'new warranty creation', 'please form a warranty for me']

sent_keys = list(dic.keys())
sentences = [warranty_listing, warranty_details, kit_details, register_claim, media_upload, media_download, warranty_creation]
sentences2 = ['warranty_listing', 'warranty_details', 'kit_details', 'register_claim', 'media_upload', 'media_download', 'warranty_creation']
type(dic)
dic = {
'warranty listing' : '',

"Search Warranty Details" : {
		  "store" : {
			"id" : "<INTEGER>",
			"distributor" : "<INTEGER"
		  },
		  "product": {
			"name": "<STRING>",
			"brand": "<STRING>",
			"category": "<STRING>",
			"serial number": "<STRING>",
			"purchase date": "<STRING>", #YYYY-mm-dd
			"invoice": "<STRING>",
			"invoice image": "<STRING-URL:>",
      "category id": "<INTEGER: product-category-id>",
      "warranty invoice" : "<STRING-URL:>",
#       ''extra info": ''<JSON>",
      "coverage":"<INTEGER>"
      },
    
		"warranty" : {
			"activation code" : "<STRING>",
			"type" : "<STRING>",
			"purchase date": "<STRING>",
			"start date": "<STRING>",
			"end date": "<STRING>",
			"duration months": "<INTEGER>",
			"price": "<FLOAT>",
			"active": "<INTEGER>", #0 or 1
			"verified": "<INTEGER>", #0 or 1 or 2
			"is zapp enable": "<INTEGER>", #is-zapp-enable
			"region_id" : "<STRING>",
			"coupon code" : "<STRING>",
			"rm id": "<INTEGER>",
			"extra info": "<JSON>",
			"endorsement no": "<STRING>",
      "oem id": "<INTEGER>",
			"cancellation at": "<STRING>",
			"cacellation done by": "<STRING>",
			"discrepancy reason": "<STRING>",
			"cancellation reason": "<STRING>",
			"cancellation role": "<STRING>", 
			"serial number": "<STRING>",
			"status": "<INTEGER>", 
		  "store assigned at" :"<STRING>",
	    "distributor assigned at" :"<STRING>",
		  "brand warranty duration months": "<INTEGER>"
		},
		"customer": {
			"name": "<STRING>",
			"email": "<STRING>",
			"phone contact": "<STRING>",
			"city": "<STRING>",
			"state": "<STRING>",
			"address": "<STRING>",
			"extrainfo": "<JSON>"
		},

		"Insured" : {
			"name": "<STRING>",
			"premium": "<STRING>",
			"contact": "<STRING>",
			"assigned date": "<STRING>",
			"state": "<STRING>"
		}

	},
"Kit Detail" :{
	"status": "<INTEGER>",
	"discrepant reason": "<STRING>",
	"purchase date": "<DATE>",
	"sold value total premium amount": "<FLOAT>",
	"sold by": "<INTEGER>",
	"duration months": "<INTEGER>",
	"distributor id" : "<INTEGER>",
	"store id": "<INTEGER>",
	"customer": {
		"name": "<STRING>",
		"phone": "<STRING>",
		"email": "<STRING>",
		"address": "<STRING>",
		"city" : "<INTEGER>",
		"pincode": "<INTEGER>"
	}},
"Register your Claim" : {
    "warranty" : "<Integer>",
    "item"     :  "<Integer>",
    "claim type" : "<Integer>",
    "concern"   :  "<STRING>",
    "voice url" :   "<String>",
    "phone"      :  "<String>",
    "address"     : "<String>",
    "remarks"      :  "<String>",
    "appointment"   :   "<Integer>",
    "extra info"     :   "<Json>",
	  "product": 	{
		"category id": "<INTEGER>",
		"brand": "<StringINTEGER:brand-id/name>",
		"model": "<INTEGER:model-idSTRING: model-name>",			
		"purchase date": "<DATE>",
		"serial number": "<STRING>",
		"invoice image": "<IMAGE>",
    "mime type": "<IMAGE>",
		"invoice number": "<STRING>",
		"product price": "<FLOAT>",
	},
	"payment": {
		"transactionid": "<STRING>"
	}
},

'media upload' : '',

'media download' : '' ,

'create new warranty' : {
	"premium": "<FLOAT>", 
	"duration months": "<INTEGER>", 
	"store" : "<INTEGER>",
	"extra info" : "<JSON: extra-informations OPTIONAL>",
	"purchased on": "<DATE>",
	"activation code" : "<STRING OPTIONAL>",
	"seller": "<STRING: name_of_warranty_seller OPTIONAL>",
  "extra images": {
		"<image desc>": "<STRING IMAGE-reference>", 
	},
	"customer": {
		"name": "<STRING>",
		"phone": "<STRING>",
		"email": "<STRING>",
		"address": "<STRING>",
		"city": "<INTEGER>",
		"pincode": "<INTEGER>"
	},
	"product": 	{
    "category": "<INTEGER>",	
		"brand": "<STRING>",
		"model": "<STRING>",
		"purchase date": "<DATE>",
		"serial number": "<STRING:serial-number>",
		"invoice image": "<IMAGE:invoice-image>",
		"invoice number": "<STRING:invoice-number>",
		"price": "<FLOAT>",
    "brand warranty duration months": "<INTEGER>" #in months		
	 }
  }
}

