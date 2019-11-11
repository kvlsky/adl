# get img labels
# labels = []
# for img_id in np.array(img_ids):
#     row = data[data['img_id'] == img_id]
#     label = row.iloc[0]['img_class']
#     labels.append(label)

# labels = tf.convert_to_tensor(labels, dtype=tf.float32)
# logits = model(images, training=True)

# counter = 0
# for sample in ds_train:
#     counter += 1
#     image = sample[0]
#     model(image)
# print(f'Batches per epoch: {counter}')

# with open('Exc_3/predictions.txt') as p1:
#     with open('Exc_3/predictions_advanced_model.txt') as p2:
#         d1 = p1.read().splitlines()
#         d2 = p2.read().splitlines()

#         for x, y in zip(d1, d2):
#             if x == y:
#                 print(x, y)
