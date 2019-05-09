# import requests
from flask import Flask, request, jsonify, make_response
app = Flask(__name__)

# @app.route("/")
# def getdata():
# 	danmu = requests.get("https://api.bilibili.com/x/v1/dm/list.so?oid=42679001")
# 	##danmu_xml = ElementTree.fromstring(danmu.content)
# 	return danmu.content
@app.route("/json",methods=['POST'])
def operate_json():
	data = request.get_json()
	id_pop = []
	content = []
	output = []
	index_c=0
	for danmu_content in data["content"]:
		if danmu_content in content:
			id_pop.append(data["content"].index(danmu_content))
			id_pop.append(index_c)
		else:
			content.append(danmu_content)
		index_c += 1
	for num in id_pop:
		if data["id"][num] not in output:
			output.append(data["id"][num])
	if 1 ==0:
		#incase anything wrong happend
		return "Error"
	else:
		return jsonify({"id":output})

if __name__ == '__main__':
	app.run(debug=True)
	#app.run(host="127.0.0.1", port="5000")