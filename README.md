# Face-Detection-System
Professional Seminar Course - Rivier University 

1. User Story
As a system administrator, I want an AI-powered face detection system that can accurately identify and locate human faces within images, so that I can enhance security and automate user authentication efficiently.
2. Description:	
A Face Detection System is created to detect faces in an image and help find them effectively for security systems, identity, and other purposes. The system will deploy some form of deep learning models including Haar Cascades and Convolutional Neural Networks (CNN). The solution will resiliently work on simple input of static images rather than the live video feed, to correctly perform face localization as well as bounding box.
3. Acceptance Criteria:
1.	Face Detection Accuracy:
•	The created system should be oriented on face detection on people’s image in specific light conditions and given positions, and its accuracy should be no less than 95%.
2.	Image Format Support:
•	The system should allow use of common image formats including JPEG, PNG, BMP among others in the exchange of images.
3.	Performance:
•	The timing of the face processing and detection should be in less than two seconds.
4.	Scalability:
•	The proposed system should also be capable of providing a batch of about one hundred images at the least while having consistently good performance.
5.	Security and Privacy:
•	The system should be capable of, undergoing data processing to eliminate identity and storing images according to data privacy laws such as GDPR.
6.	Visualization:
•	It is recommended that when faces have been detected they should be circled or boxed on the processed images.
7.	Error Handling:
•	The system should be able to give informative error messages in case no faces are detected or the input image is not valid.
8.	Logging and Reporting:
•	A record of the faces detected and time taken to detect the faces should be printed as a record of the audit.
4. Conditions of Satisfaction:
•	The system needs to identify false positives and negatives in the images, and the number should be as close to zero as possible.
•	Face detection should not only work well when dealing with clear images but also noisy ones.
•	The tool should generate good bounding boxes on the faces and the boxes should be closely cropped around the faces.
•	There should be clear messages to the users if an image is not processed properly.
5. Definition of Done:
•	The system can analyze and recognize faces in different types of images.
•	All the acceptance criteria are satisfied, and they are checked during unit and integration tests.
•	A simple interface for uploading images and checking results is provided.
•	Documentation and logs are also done properly for future use.
•	System functionality efficiency such as response time is a priority.
6. Agents Overview:
1.	Image Preprocessing Agent:
o	Ensures images are in the correct format and quality for processing.
2.	Face Detection Agent:
o	Implements deep learning models (Haar Cascades, CNNs) to detect faces.
3.	Bounding Box Generation Agent:
o	Generates bounding boxes around detected faces with precise coordinates.
4.	Result Analysis Agent:
o	Aggregates results and prepares summary reports for user review.
5.	Logging & Reporting Agent:
o	Maintains records of processed images, detection success, and performance metrics.
7.  Detailed Agent Descriptions:
1. Image Input Agent
•	Role: Handles image uploads and preprocessing.
•	Tasks:
o	Accepts images via web interface.
o	Resizes and normalizes images.
o	Ensures input validity (correct format and resolution).
•	Interactions: Works with Face Detection Agent.
2. Face Detection Agent
•	Role: Processes the image and detects faces using deep learning models.
•	Tasks:
o	Load pre-trained models (Haar Cascades, CNNs).
o	Detect faces and return bounding box coordinates.
•	Interactions: Works with Visualization Agent.
3. Visualization Agent
•	Role: Draws bounding boxes around detected faces on the image.
•	Tasks:
o	Annotate detected faces with bounding boxes.
o	Return annotated images to the web interface.
•	Interactions: Works with Image Input Agent.
4. Logging Agent
•	Role: Maintains logs of detected faces and processing times.
•	Tasks:
o	Store logs in a database.
o	Generate reports for analysis.
•	Interactions: Works with Security Compliance Agent.
5. Error Handling Agent
•	Role: Manages errors related to image processing and face detection.
•	Tasks:
o	Detect and report missing or corrupted images.
o	Provide user-friendly error messages.
•	Interactions: Works with Image Input Agent.
6. Security Compliance Agent
•	Role: Ensures compliance with privacy regulations.
•	Tasks:
o	Anonymize detected face data.
o	Ensure secure storage of processed images.
•	Interactions: Works with Logging Agent.
8. Agent Interactions:
•	The Image Preprocessing Agent processes images by preparing them and passing the images to the Face Detection Agent.
•	The purpose of the Face Detection Agent is to detect faces and deliver the coordinates to the Bounding Box Generation Agent.
•	The Bounding Box Generation Agent provides annotation to the image and then passes it on to the Result Analysis Agent.
•	The Result Analysis Agent amasses result and share data with the Logging & Reporting Agent.
•	Processed images and detection reports are requested by users from the Result Analysis Agent.
9. Workflow:
1.	Image Upload:
o	User uploads an image to the system.
2.	Preprocessing:
o	Image Preprocessing Agent converts and optimizes the image.
3.	Face Detection:
o	Face Detection Agent applies deep learning models to detect faces.
4.	Bounding Box Drawing:
o	Bounding Box Generation Agent marks detected faces on the image.
5.	Result Analysis:
o	Results are analyzed, and detection summary is generated.
6.	Logging & Reporting:
o	Detection data is logged, and reports are provided to the user.
7.	Output Display:
o	The processed image with bounding boxes is displayed to the user.
10. Tasks
US11.1: Implement Image Input Handling (12 ph) #201
Agent: Image Input Agent
•	Subtask 1: The Image Input Agent should parse and manage image files uploaded by the user, should check multiple formats (JPEG, PNG). (5 ph)
•	Subtask 2: Another is to incorporate validation checks to validate image quality, its size, and resolution required. (4 ph)
•	Subtask 3: Treat corrupted or unsupported image files with error messages the right way. (3 ph)
US11.2: Implement Face Detection Module (20 ph) #202
Agent: Face Detection Agent
•	Subtask 1: The key face detection functionality should be established with Haar cascades and CNN models to detect faces in images. (8 ph)
•	Subtask 2: Increase accuracy and decrease false positives/negatives of detection so that it can be extremely fast. (6 ph)
•	Subtask 3: Continue to support the detection of multiple faces in a single picture using bounding boxes. (4 ph)
•	Subtask 4: Using another set of images from a different population sample, confirm the degree of accuracy in detection. (2 ph)
US11.3: Implement Preprocessing Module (16 ph) #203
Agent: Preprocessing Agent
•	Subtask 1: Define a process of image preprocessing which consist of resizing, conversion to grayscale and normalization. (6 ph)
•	Subtask 2: Adjust image parameters to achieve better results of face detection, increasing contrast and reducing noise. (5 ph)
•	Subtask 3: Test the achieved preprocessing steps with the aim not to worsen the detection performance. (5 ph)
US11.4: Develop Results Visualization Module (14 ph) #204
Agent: Visualization Agent
•	Subtask 1: Add visualization components to place bounding boxes on acknowledged faces and show results to users. (6 ph)
•	Subtask 2: Offer opportunities to download the final images with face markings in additional formats. (4 ph)
•	Subtask 3: Make sure the visualization module should be responsive and supported multiple resolutions. (4 ph)
US11.5: Develop Performance Optimization (18 ph) #205
Agent: Performance Optimization Agent
•	Subtask 1: Improve the system performance and accuracy in large datasets containing images without much delay. (6 ph)
•	Subtask 2: Introduce use of multithreading to enhance performance of the program and decrease response time. (6 ph)
•	Subtask 3: Establishing benchmarks in terms of perfomance at various loads and adjusting performances thereto. (6 ph)
US11.6: Implement Error Handling and Logging (10 ph) #206
Agent: Error Handling Agent
•	Subtask 1: Log error messages and other types of messages during image processing implementation. (4 ph)
•	Subtask 2: Help users to identify issues by offering helpful error messages to enable them solve problems easily. (4p)
•	Subtask 3: Make sure logs are properly archived and can then accessed in case of a bug identification. (2 ph)
US11.7: Develop Security Features (12 ph) #207
Agent: Security Agent
•	Subtask 1: To enhance security during data transfer, encryption should be introduced to the image uploads, as well as the results that the system has processed. (6 ph)
•	Subtask 2: Compliance with other set data privacy laws such as the General Data Protection Regulation (GDPR). (4 ph)
•	Subtask 3: Ensure the unauthorized use of the devices by integrating positive user access control features. (2 ph)
US11.8: Conduct System Testing and Validation (16 ph) #208
Agents Involved: All Agents
•	Subtask 1: Run various image sets end to end and test the entire system to determine its functionality. (6 ph)
•	Subtask 2: Strengthen the argument that Face Detection Agent yields accurate results each time it is run. (6 ph)
•	Subtask 3: Check that interaction between all modules works as expected and implements the necessary functionality for the chosen scenarios. (4 ph)
