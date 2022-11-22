from thresholding import main
from flask import Flask, flask_request

path = './data/22.jpg'

app = Flask(__name__) 
 
@app.route("/result", methods = ["POST", "GET"]) 
def result(): 
    output  = flask_request.get_json() 
     
    link = output['resumelink'] 
    if link:
        main(link)
 
if __name__ == '__main__': 
    app.run(debug=True, port=2000)