import argparse
from metrics import sum_entropy, max_entropy, mutual_information, KL_score
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--method', '-m',
                    help='Active learning method. Choose one: sum_entropy, max_entropy, mutual_information or KL_score',
                    required=True)
parser.add_argument('--visualize', '-v',
                    help='Flag to visualize selected image',
                    default=True) 
parser.add_argument('--pred_path', '-p',
                    help='Paths to prediction results. Eg: ./pred1.json,./pred2.json',
                    required=True)
parser.add_argument('--num_query', '-n',
                    help='Number of selected images',
                    default=10)
args = parser.parse_args()

method = args.method
visualize = args.visualize
pred_path = args.pred_path.split(',')
num_query = args.num_query


def method_selection(method):
    '''
    Given name of method, return function
    '''
    switcher = {
        'max_entropy'       : max_entropy,
        'sum_entropy'       : sum_entropy,
        'mutual_information': mutual_information,
        'KL_score'          : KL_score
    }

    return switcher.get(method, 'Invalid query method')


def main():
    if method in ['sum_entropy', 'max_entropy']:
        assert len(pred_path) == 1, 'sum_entropy and max_entropy require prediction results from a single model'
    elif method in ['mutual_information', 'KL_score']:
        assert len(pred_path) > 1, 'mutual_information and KL_score require prediction results from at least two models'
    else:
        raise ValueError('Invalid query method')

    query_method =  method_selection(method)
    pred_dict_list = utils.create_im_dict_from_path(pred_path)

    uncertainty_score = query_method(pred_dict_list)
    selected_images = uncertainty_score.argsort()[0][::-1][:num_query]
    image_ids = list(pred_dict_list[0].keys())
    image_ids.sort()
    print('Selected images for query:', [image_ids[i] for i in selected_images])

    #Need to fix visualization method for predictions with no groundtruths.
    # Visualization
    if visualize:
        for i in selected_images:
            image_id = image_ids[i]
            img = utils.bbox_visualize(image_id, img_root='../TFS_analyze/test_data_20210524/data/')
            img.show()


if __name__ == '__main__':
    main()