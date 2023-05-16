import data_reader as dr
import tensorflow as tf



drs = dr.data_reader("outputOfPhase2Training.csv")
dataset = drs.createTFDataset()

drs_test = dr.data_reader("outputOfPhase2Test.csv")
dataset_test = drs_test.createTFDataset(upSample=True)

dataset.stats_generator()
dataset_test.stats_generator()


print("There are ", dataset.cardinality().numpy(), " samples in the dataset")
print("Each sample has a shape of ", dataset.element_spec[0].shape)
print("\n")
print("There are ", dataset_test.cardinality().numpy(), " samples in the test dataset")
print("Each sample has a shape of ", dataset_test.element_spec[0].shape)

# create a model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name="Conv2D_1"),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name="Conv2D_2"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])



model.compile(optimizer=tf.keras.optimizers.Adadelta(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.build(input_shape=(None, 15, 15, 1))
print(model.summary())

dataset = dataset

for i in dataset.take(1):
    print(i[0].shape)
    print(i[1])

model.fit(dataset, epochs=200, verbose=1)

prediction = model.predict(dataset_test)
print(model.evaluate(dataset_test, verbose=1))



model.save("model.h5")












