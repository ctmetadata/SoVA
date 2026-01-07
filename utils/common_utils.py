import json

# from config.graphrag_prompt import TEST_FILE_PATH, TEST_GROUNDTRUTH_PATH

# test_file_handle = open(TEST_FILE_PATH, encoding='utf-8')
# test_file_content = json.load(test_file_handle)
# test_groundtruth_content = open(TEST_GROUNDTRUTH_PATH, encoding='utf-8').readlines()
# test_groundtruth_jsonline = list(map(lambda x: json.loads(x), test_groundtruth_content))


# def get_test_info(test_id):
#     target = list(filter(lambda x: x["id"] == str(test_id), test_file_content))[0]
#     return target["id"], target["input"]


# def get_ground_truth(test_id):
#     target = list(filter(lambda x: x["id"] == str(test_id), test_groundtruth_jsonline))[0]
#     return target["output"]


# if __name__ == '__main__':
#     test_id = "131406"
#     get_ground_truth(test_id)
