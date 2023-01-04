from flask import Flask, request
from thresholding import main

app = Flask(__name__) 
 
@app.route("/result", methods = ["POST"]) 
def result(): 
    link, filetype, res = None, None, None
    output  = request.get_json()
    link = output['resumelink']
    filetype = output['idtype']

    if link and filetype:
        res = main(link, filetype)
        return res

if __name__ == '__main__':
    app.run(debug=True, port=8000)