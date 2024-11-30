# project_aria.py
import os
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId
from tqdm import tqdm

"""
Sensor 	Stream ID 	Recordable Type ID 	label
ET camera 	211-1 	EyeCameraRecordableClass 	camera-et
RGB camera 	214-1 	RgbCameraRecordableClass 	camera-rgb
Microphone 	231-1 	StereoAudioRecordableClass 	mic
Barometer 	247-1 	BarometerRecordableClass 	baro
GPS 	281-1 	GpsRecordableClass 	gps
Wi-Fi 	282-1 	WifiBeaconRecordableClass	wps
Bluetooth 	283-1 	BluetoothBeaconRecordableClass	bluetooth
SLAM/Mono Scene camera left 	1201-1 	SlamCameraData 	camera-slam-left
SLAM/Mono Scene camera right	1201-2 	SlamCameraData 	camera-slam-right
IMU (1kHz) 	1202-1 	SlamImuData 	imu-right
IMU (800Hz) 	1202-2 	SlamImuData 	imu-left
Magnetometer 	1203-1 	SlamMagnetometerData 	mag 
"""

class AriaDataProvider:
	def __init__(self, file_path):
		self.vrs_path = file_path
		self.provider = data_provider.create_vrs_data_provider(file_path)
		if not self.provider:
			raise ValueError("Invalid VRS data provider")

	def get_image_by_index(self, stream_id, index):
		return self.provider.get_image_data_by_index(stream_id, index)[0].to_numpy_array()

	def get_image_by_time_ns(self, stream_id, time_ns, time_domain=TimeDomain.DEVICE_TIME, option=TimeQueryOptions.CLOSEST):
		return self.provider.get_image_data_by_time_ns(stream_id, time_ns, time_domain, option)[0].to_numpy_array()

	def get_stream_id_from_label(self, label):
		return self.provider.get_stream_id_from_label(label)

	def get_stream_label_from_id(self,id):
		return self.provider.get_label_from_stream_id(id)

	def get_calibration(self):
		return self.provider.get_device_calibration()

	def summarize_vrs(self, rgb_stream_id=StreamId("214-1"), num_thumbnails=10, resize_ratio=10):
	
		# Retrieve image size for the RGB stream
		time_domain = TimeDomain.DEVICE_TIME  # query data based on host time
		option = TimeQueryOptions.CLOSEST # get data whose time [in TimeDomain] is CLOSEST to query time

		# Retrieve Start and End time for the given Sensor Stream Id
		start_time = self.provider.get_first_time_ns(rgb_stream_id, time_domain)
		end_time = self.provider.get_last_time_ns(rgb_stream_id, time_domain)

		image_config = self.provider.get_image_configuration(rgb_stream_id)
		width = image_config.image_width
		height = image_config.image_height

		sample_count = 10
		resize_ratio = 10
		thumbnail = newImage = Image.new(
			"RGB", (int(width * sample_count / resize_ratio), int(height / resize_ratio))
		)
		current_width = 0


		# Samples 10 timestamps
		sample_timestamps = np.linspace(start_time, end_time, sample_count)
		for sample in tqdm(sample_timestamps):
			image_tuple = self.provider.get_image_data_by_time_ns(rgb_stream_id, int(sample), time_domain, option)
			image_array = image_tuple[0].to_numpy_array()
			image = Image.fromarray(image_array)
			new_size = (
				int(image.size[0] / resize_ratio),
				int(image.size[1] / resize_ratio),
			)
			image = image.resize(new_size).rotate(-90)
			thumbnail.paste(image, (current_width, 0))
			current_width = int(current_width + width / resize_ratio)
		return thumbnail
	
	def get_sensor_data_by_index(self):
		# # Random access data
		# Goal
		# - Access data from a stream randomly using a data index or a timestamp
		# 
		# Key learnings
		# - Sensor data can be obtained through index within the range of [0, number of data for this stream_id)
		# 
		#   - `get_sensor_data_by_index(stream_id, index)`
		#   - `get_image_data_by_index(stream_id, index)`
		#   - Access other sensor data by index interface is available in core/python/VrsDataProviderPyBind.h
		#   
		# - `TimeQueryOptions` has three options: `TimeQueryOptions.BEFORE`, `TimeQueryOptions.AFTER`, `TimeQueryOptions.CLOSEST`
		# - Query through index will provide the exact data vs query through a timestamp that is not exact, data nearby will be omitted base on `TimeQueryOptions`

		sensor_name = "camera-slam-right"
		sensor_stream_id = self.provider.get_stream_id_from_label(sensor_name)

		# get all image data by index
		num_data = self.provider.get_num_data(sensor_stream_id)

		for index in range(0, num_data):
			image_data = self.provider.get_image_data_by_index(sensor_stream_id, index)
			print(
				f"Get image: {index} with timestamp {image_data[1].capture_timestamp_ns}"
			)

	def get_sensor_data_by_time_ns(self):
		# ### Sensor data can be obtained by timestamp (nanoseconds)
		# * Get stream time range `get_first_time_ns` and `get_last_time_ns`
		# * Specify timedomain: `TimeDomain.DEVICE_TIME` (default)
		# * Query data by queryTime
		#   * `TimeQueryOptions.BEFORE` (default): sensor_dataTime <= queryTime
		#   * `TimeQueryOptions.AFTER` : sensor_dataTime >= queryTime
		#   * `TimeQueryOptions.CLOSEST` : sensor_dataTime closest to queryTime


		sensor_name = "camera-slam-right"
		sensor_stream_id = self.provider.get_stream_id_from_label(sensor_name)

		time_domain = TimeDomain.DEVICE_TIME  # query data based on DEVICE_TIME
		option = TimeQueryOptions.CLOSEST # get data whose time [in TimeDomain] is CLOSEST to query time

		start_time = self.provider.get_first_time_ns(sensor_stream_id, time_domain)
		end_time = self.provider.get_last_time_ns(sensor_stream_id, time_domain)

		for time in range(start_time, end_time, int(1e7)):
			image_data = self.provider.get_image_data_by_time_ns(
				sensor_stream_id, time, time_domain, option
			)
			print(
				f"query time {time} and get capture image time {image_data[1].capture_timestamp_ns} within range {start_time} {end_time}"
			)

	@staticmethod
	def sensor_config_console_output(config):
		print(f"device_type {config.device_type}")
		print(f"device_version {config.device_version}")
		print(f"device_serial {config.device_serial}")
		print(f"sensor_serial {config.sensor_serial}")
		print(f"nominal_rate_hz {config.nominal_rate_hz}")
		print(f"image_width {config.image_width}")
		print(f"image_height {config.image_height}")
		print(f"pixel_format {config.pixel_format}")

	def get_sensor_data_config(self):

		sensor_name = "camera-slam-right"
		sensor_stream_id = self.provider.get_stream_id_from_label(sensor_name)
		config = self.provider.get_image_configuration(sensor_stream_id)
		AriaDataProvider.sensor_config_console_output(config)

		
	def get_sensor_data_w_capture_time(self):
		# # Get sensor data in a sequence based on data capture time
		# Goal:
		# - Obtain sensor data sequentially based on timestamp
		# 
		# Key learnings
		# - Default option activates all sensors and playback the entire dataset from vrs
		# - Setup option to only activate certain streams, truncate start/end time, and sample rate
		# - Obtain data from different sensor types
		# - `TimeDomain` are separated into four categories: `RECORD_TIME`, `DEVICE_TIME`, `HOST_TIME`, `TIME_CODE`

		# ### Step 1: obtain default options that provides the whole dataset from VRS
		# * activates all sensor streams
		# * No truncation for first/last timestamp
		# * Subsample rate = 1 (do not skip any data per sensor)

		options = (
			self.provider.get_default_deliver_queued_options()
		)  # default options activates all streams

		# ### Step 2: set preferred deliver options
		# * truncate first/last time: `set_truncate_first_device_time_ns/set_truncate_last_device_time_ns()`
		# * subselect sensor streams to play: `activate_stream(stream_id)`
		# * skip sensor data : `set_subsample_rate(stream_id, rate)`

		options.set_truncate_first_device_time_ns(int(1e8))  # 0.1 secs after vrs first timestamp
		options.set_truncate_last_device_time_ns(int(1e9))  # 1 sec before vrs last timestamp

		# deactivate all sensors
		options.deactivate_stream_all()
		# activate only a subset of sensors
		slam_stream_ids = options.get_stream_ids(RecordableTypeId.SLAM_CAMERA_DATA)
		imu_stream_ids = options.get_stream_ids(RecordableTypeId.SLAM_IMU_DATA)

		for stream_id in slam_stream_ids:
			options.activate_stream(stream_id)  # activate slam cameras
			options.set_subsample_rate(stream_id, 1)  # sample every data for each slam camera
			
		for stream_id in imu_stream_ids:
			options.activate_stream(stream_id)  # activate imus
			options.set_subsample_rate(stream_id, 10)  # sample every 10th data for each imu

		# ### Step 3: create iterator to deliver data
		# `TimeDomain` contains the following
		# * `RECORD_TIME`: timestamp stored in vrs index, fast to access, but not guaranteed which time domain
		# * `DEVICE_TIME`: capture time in device's timedomain, accurate
		# * `HOST_TIME`: arrival time in host computer's timedomain, may not be accurate
		# * `TIME_CODE`: capture in TimeSync server's timedomain
		# 

		iterator = self.provider.deliver_queued_sensor_data(options)
		for sensor_data in iterator:
			label = self.provider.get_label_from_stream_id(sensor_data.stream_id())
			sensor_type = sensor_data.sensor_data_type()
			device_timestamp = sensor_data.get_time_ns(TimeDomain.DEVICE_TIME)
			host_timestamp = sensor_data.get_time_ns(TimeDomain.HOST_TIME)
			timecode_timestamp = sensor_data.get_time_ns(TimeDomain.TIME_CODE)
			print(
				f"""obtain data from {label} of type {sensor_type} with \n
				DEVICE_TIME: {device_timestamp} nanoseconds \n
				HOST_TIME: {host_timestamp} nanoseconds \n
				"""
			)


# Example Usage
def example_usage():
	file_path ="/mnt/homes/minghao/robotflow/egoego/third_party/projectaria_tools/data/mps_sample/sample.vrs"
	provider = AriaDataProvider(file_path)
	rgb_stream_id = provider.get_stream_id_from_label("camera-rgb")
	image = provider.get_image_by_index(rgb_stream_id, 1)
	print("Image data retrieved:", image.shape)

if __name__ == "__main__":
	example_usage()
