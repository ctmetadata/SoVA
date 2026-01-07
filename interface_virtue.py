import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) 

from utils.customgraphrag_virtue import GraphragV2
from flask import Flask, request, jsonify
from typing import List, Dict, Optional
import json
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"  
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

rag_instance = None
is_rag_init = False

def init_rag_instance():
    global rag_instance
    try:
        rag_instance = GraphragV2(
            INPUT_DIR=""
        )
        logger.info("✅ GraphragV2 instance initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize GraphragV2 instance: {str(e)}", exc_info=True)
        return False

@app.before_request
def initialize():
    global rag_instance, is_rag_init  
    if not is_rag_init:  
        init_success = init_rag_instance()
        if not init_success:
            raise RuntimeError("GraphragV2 initialization failed, service cannot start")
        is_rag_init = True  

@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Received health check request")
    return jsonify({
        "status": "healthy",
        "service": "graphrag-flask-api",
        "version": "1.0.0",
        "time": json.dumps(str(datetime.now()), ensure_ascii=False)  
    })

@app.route('/graphrag_result', methods=['POST'])
def graphrag_query():
    try:
        try:
            request_data = request.get_json()
            logger.info(f"Received request data | Type: {type(request_data)} | Content: {json.dumps(request_data, ensure_ascii=False)}")
        except Exception as e:
            logger.error(f"Failed to parse request JSON: {str(e)}", exc_info=True)
            return jsonify({
                "status": "error",
                "final_answer": "Request format is not valid JSON",
                "error_detail": str(e)
            }), 400

        if not request_data:
            logger.warning("Request data is empty (None)")
            return jsonify({
                "status": "error",
                "final_answer": "Request data is empty, please send JSON format data"
            }), 400

        messages = request_data.get("message", [])
        logger.info(f"Extracted messages | Type: {type(messages)} | Length: {len(messages)} | Content: {json.dumps(messages, ensure_ascii=False)}")

        if not isinstance(messages, list):
            logger.error(f"messages is not a list | Actual type: {type(messages)}")
            return jsonify({
                "status": "error",
                "final_answer": "messages must be in list format",
                "error_detail": f"Actual type: {type(messages)}"
            }), 400
        if len(messages) == 0:
            logger.warning("messages list is empty")
            return jsonify({
                "status": "error",
                "final_answer": "messages list cannot be empty, must contain at least one message"
            }), 400

        last_msg = messages[-1]
        if not isinstance(last_msg, dict):
            logger.error(f"Last message is not a dictionary | Type: {type(last_msg)} | Content: {last_msg}")
            return jsonify({
                "status": "error",
                "final_answer": "Elements in messages must be dictionary format (e.g., {\"role\":\"user\",\"content\":\"question\"})"
            }), 400

        content = last_msg.get("content", "").strip()
        logger.info(f"Extracted content from last message | Original value: {last_msg.get('content')} | Cleaned: {content} | Length: {len(content)}")

        if not content:
            logger.warning(f"User question is empty | Cleaned content: {content} | Length: {len(content)}")
            return jsonify({
                "status": "failed",
                "error": "User question cannot be empty",
                "final_answer": "",
                "debug_info": f"Last message: {json.dumps(last_msg, ensure_ascii=False)}"
            }), 400

        threshold = request_data.get("threshold", 60)
        topN = request_data.get("topN", 5)
        logger.info(f"Extracted parameters | threshold: {threshold} | topN: {topN}")

        try:
            logger.info("Starting GraphragV2 request processing")
            graphrag_result = rag_instance(
                messages=messages,
                threshold=threshold,
                topN=topN
            )

            response = graphrag_result["final_answer"]
            first_prompt = graphrag_result["search_output"]["data"]["prompts"]["first_prompt"]
            second_prompt = graphrag_result["search_output"]["data"]["prompts"]["second_prompt"]
            logger.info(f"GraphragV2 call successful | Answer length: {len(response)}")
            return jsonify(response)
        except Exception as e:
            logger.error(f"GraphragV2 call failed: {str(e)}", exc_info=True)
            return jsonify({
                "status": "error",
                "final_answer": f"GraphragV2 processing failed: {str(e)}",
                "error_detail": str(e)
            }), 500

    except Exception as e:
        logger.error(f"Interface processing exception: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "final_answer": f"Interface processing failed: {str(e)}",
            "error_detail": str(e)
        }), 500

if __name__ == '__main__':
    import argparse
    import datetime  
    parser = argparse.ArgumentParser(description="GraphRAG Flask API Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Service binding IP address")
    parser.add_argument("--port", type=int, default=8501, help="Service listening port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    logger.info(f"Starting GraphRAG API service | host: {args.host} | port: {args.port} | debug: {args.debug}")
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )