from thresholding import main
from flask import Flask, request

path = './data/22.jpg'
app = Flask(__name__) 
 
@app.route("/result", methods = ["POST", "GET"]) 
def result(): 
    output  = request.get_json() 
     
    link = output['resumelink']
    filetype = output['idtype']

    if link and filetype:
        main(link, filetype)
 
if __name__ == '__main__': 
    app.run(debug=True, port=2000)