# from flask import Flask, request
from thresholding import main

# app = Flask(__name__) 
 
# @app.route("/result", methods = ["POST"]) 
# def result(): 
    # link, filetype, res = None, None, None
    # output  = request.get_json()
    # link = output['resumelink']
    # filetype = output['idtype']
link = 'https://elma365dev.technodom.kz/s3elma365/af209a06-169d-44cb-b849-8b9459cb4d74?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=PZSF73JG72Ksd955JKU1HIA%2F20221228%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20221228T115447Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&response-content-disposition=inline&X-Amz-Signature=1b030f01f9322af9a36a8f53075352b354843d791f3588aa9ed8e8afb49bfcd8'
filetype = 'pdf'

if link and filetype:
    res = main(link, filetype)
    # return res
    print(res)

# if __name__ == '__main__':
    # app.run(debug=True, port=8000)