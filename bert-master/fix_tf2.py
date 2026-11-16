import os

files_to_fix = ['modeling.py', 'optimization.py', 'tokenization.py', 'run_classifier.py', 'extract_features.py', 'run_pretraining.py', 'run_squad.py', 'create_pretraining_data.py']

tf_compat_code = """import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
"""

for filename in files_to_fix:
    if not os.path.exists(filename):
        continue
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # 如果已经修改过，跳过
    if 'tf.disable_v2_behavior()' in content:
        print(f'{filename} already fixed, skipping...')
        continue
    
    # 在 import tensorflow as tf 后面添加兼容代码
    if 'import tensorflow as tf' in content:
        content = content.replace(
            'import tensorflow as tf',
            'import tensorflow as tf\n' + tf_compat_code
        )
        
        with open(filename, 'w') as f:
            f.write(content)
        print(f'Fixed {filename}')
    else:
        print(f'{filename} does not contain tensorflow import')

print('Done!')
