import json
import sys
import tomllib
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


IMAGE = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "calibration/captures/shelf_markers.jpg"
)

LAYOUT = Path(
    "calibration/shelf_markers.toml"
)

CAMERA_POSE = Path(
    "calibration/camera_pose_bootstrap.json"
)

MODEL = Path("models/best.pt")

ROI_OUTPUT = Path(
    "calibration/captures/calibration_roi.jpg"
)

ANNOTATED_OUTPUT = Path(
    "calibration/captures/calibration_roi_coordinates.jpg"
)

JSON_OUTPUT = Path(
    "calibration/monocular_roi_coordinates.json"
)


MARKER_IDS = (20, 21, 22, 23)

CONFIDENCE = 0.30
YOLO_IMAGE_SIZE = 960
PIXELS_PER_MM = 4.0

# Empty means all detected classes.
# Use "Ball" to ignore every other class.
ONLY_CLASS = ""


# Physical shelf surfaces derived from:
#
# first compartment begins at Y=125 mm
# compartment height = 120 mm
# shelf thickness = 5 mm
#
# The first usable bottom surface is therefore Y=245 mm.
SHELF_SURFACE_Y_MM = (
    245.0,
    370.0,
    495.0,
)


def world_ray(
    pixel,
    camera_matrix,
    distortion,
    rotation_world_to_camera,
):
    corrected = cv2.undistortPoints(
        np.asarray(
            pixel,
            dtype=np.float64,
        ).reshape(1, 1, 2),
        camera_matrix,
        distortion,
    ).reshape(2)

    camera_ray = np.array(
        [
            corrected[0],
            corrected[1],
            1.0,
        ],
        dtype=np.float64,
    )

    ray = (
        rotation_world_to_camera.T
        @ camera_ray
    )

    return ray / np.linalg.norm(ray)


def intersect_horizontal_shelf(
    camera_position,
    ray,
    shelf_y_mm,
):
    denominator = float(ray[1])

    if abs(denominator) < 1e-9:
        raise RuntimeError(
            "Camera ray is parallel to shelf"
        )

    distance = (
        float(shelf_y_mm)
        - float(camera_position[1])
    ) / denominator

    if distance <= 0:
        raise RuntimeError(
            "Shelf intersection is behind camera"
        )

    return (
        camera_position
        + distance * ray
    )


image = cv2.imread(str(IMAGE))

if image is None:
    raise FileNotFoundError(
        f"Could not read image: {IMAGE}"
    )


with LAYOUT.open("rb") as stream:
    layout = tomllib.load(stream)


pose = json.loads(
    CAMERA_POSE.read_text(
        encoding="utf-8"
    )
)


image_height, image_width = (
    image.shape[:2]
)

calibration_width = int(
    pose["image_size"]["width"]
)

calibration_height = int(
    pose["image_size"]["height"]
)

if (
    image_width,
    image_height,
) != (
    calibration_width,
    calibration_height,
):
    raise RuntimeError(
        "Image resolution does not match "
        "camera calibration: "
        f"image={image_width}x{image_height}, "
        f"calibration="
        f"{calibration_width}x{calibration_height}"
    )


camera_matrix = np.asarray(
    pose["camera_matrix"],
    dtype=np.float64,
)

distortion = np.asarray(
    pose["distortion_coefficients"],
    dtype=np.float64,
).reshape(-1, 1)

rotation = np.asarray(
    pose["rotation_world_to_camera"],
    dtype=np.float64,
)

translation = np.asarray(
    pose["translation_world_to_camera_mm"],
    dtype=np.float64,
).reshape(3)

camera_position = (
    -rotation.T
    @ translation
)


dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

parameters = cv2.aruco.DetectorParameters()

parameters.cornerRefinementMethod = (
    cv2.aruco.CORNER_REFINE_SUBPIX
)

detector = cv2.aruco.ArucoDetector(
    dictionary,
    parameters,
)

corners, ids, _ = detector.detectMarkers(
    image
)


detected = {}

if ids is not None:
    for marker_corners, marker_id in zip(
        corners,
        ids.flatten(),
    ):
        marker_id = int(marker_id)

        if marker_id in MARKER_IDS:
            detected[marker_id] = (
                marker_corners.reshape(
                    4,
                    2,
                )
            )


missing = set(MARKER_IDS) - set(detected)

if missing:
    raise RuntimeError(
        f"Missing shelf markers: "
        f"{sorted(missing)}"
    )


marker_size_mm = float(
    layout["aruco"][
        "shelf_marker_size_mm"
    ]
)

half_marker_mm = (
    marker_size_mm / 2.0
)


marker_centres = {
    marker_id: [
        float(value)
        for value in layout["markers"][
            str(marker_id)
        ]["center_mm"]
    ]
    for marker_id in MARKER_IDS
}


z_min_mm = min(
    marker_centres[20][2],
    marker_centres[22][2],
) - half_marker_mm

z_max_mm = max(
    marker_centres[21][2],
    marker_centres[23][2],
) + half_marker_mm

y_min_mm = min(
    marker_centres[20][1],
    marker_centres[21][1],
) - half_marker_mm

y_max_mm = max(
    marker_centres[22][1],
    marker_centres[23][1],
) + half_marker_mm


roi_width_mm = (
    z_max_mm - z_min_mm
)

roi_height_mm = (
    y_max_mm - y_min_mm
)

roi_width_px = int(
    round(
        roi_width_mm
        * PIXELS_PER_MM
    )
)

roi_height_px = int(
    round(
        roi_height_mm
        * PIXELS_PER_MM
    )
)


# Outside corners of the four markers:
# top-left, top-right, bottom-right, bottom-left.
source_quad = np.float32(
    [
        detected[20][0],
        detected[21][1],
        detected[23][2],
        detected[22][3],
    ]
)

target_quad = np.float32(
    [
        [0, 0],

        [
            roi_width_px - 1,
            0,
        ],

        [
            roi_width_px - 1,
            roi_height_px - 1,
        ],

        [
            0,
            roi_height_px - 1,
        ],
    ]
)


image_to_roi = cv2.getPerspectiveTransform(
    source_quad,
    target_quad,
)


# No horizontal shifting is performed.
# Therefore all four calibration markers remain visible.
roi = cv2.warpPerspective(
    image,
    image_to_roi,
    (
        roi_width_px,
        roi_height_px,
    ),
    flags=cv2.INTER_CUBIC,
)


ROI_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

if not cv2.imwrite(
    str(ROI_OUTPUT),
    roi,
):
    raise RuntimeError(
        f"Could not save ROI: {ROI_OUTPUT}"
    )


model = YOLO(str(MODEL))

prediction = model.predict(
    source=image,
    imgsz=YOLO_IMAGE_SIZE,
    conf=CONFIDENCE,
    iou=0.45,
    verbose=False,
)[0]


objects = []


if prediction.boxes is not None:
    for box, class_id, score in zip(
        prediction.boxes.xyxy.cpu().numpy(),
        prediction.boxes.cls.cpu().numpy(),
        prediction.boxes.conf.cpu().numpy(),
    ):
        class_name = str(
            model.names[int(class_id)]
        )

        if (
            ONLY_CLASS
            and class_name.casefold()
            != ONLY_CLASS.casefold()
        ):
            continue

        x1, y1, x2, y2 = (
            float(value)
            for value in box
        )

        centre_pixel = np.array(
            [
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
            ],
            dtype=np.float64,
        )

        bottom_pixel = np.array(
            [
                (x1 + x2) / 2.0,
                y2,
            ],
            dtype=np.float64,
        )


        roi_centre = (
            cv2.perspectiveTransform(
                centre_pixel.reshape(
                    1,
                    1,
                    2,
                ),
                image_to_roi,
            ).reshape(2)
        )


        # Ignore detections outside the calibration-marker ROI.
        if not (
            0
            <= roi_centre[0]
            < roi_width_px

            and

            0
            <= roi_centre[1]
            < roi_height_px
        ):
            continue


        roi_bottom = (
            cv2.perspectiveTransform(
                bottom_pixel.reshape(
                    1,
                    1,
                    2,
                ),
                image_to_roi,
            ).reshape(2)
        )


        provisional_bottom_y_mm = (
            y_min_mm
            + float(roi_bottom[1])
            * roi_height_mm
            / (roi_height_px - 1)
        )


        # Select the physically closest measured shelf.
        selected_shelf_y_mm = min(
            SHELF_SURFACE_Y_MM,
            key=lambda shelf_y: abs(
                shelf_y
                - provisional_bottom_y_mm
            ),
        )


        # Ray through the bottom of the detected object.
        bottom_ray = world_ray(
            bottom_pixel,
            camera_matrix,
            distortion,
            rotation,
        )


        # Physical contact point where the ray reaches
        # the measured horizontal shelf.
        contact_point = (
            intersect_horizontal_shelf(
                camera_position,
                bottom_ray,
                selected_shelf_y_mm,
            )
        )


        # Ray through the centre of the detected object.
        centre_ray = world_ray(
            centre_pixel,
            camera_matrix,
            distortion,
            rotation,
        )


        if abs(
            float(centre_ray[0])
        ) < 1e-9:
            raise RuntimeError(
                "Object-centre ray is parallel "
                "to the X plane"
            )


        # Put the object centre at the same physical
        # X depth as its shelf contact point.
        centre_distance = (
            float(contact_point[0])
            - float(camera_position[0])
        ) / float(centre_ray[0])


        centre_point = (
            camera_position
            + centre_distance
            * centre_ray
        )


        source_box = np.array(
            [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ],
            dtype=np.float64,
        ).reshape(-1, 1, 2)


        roi_polygon = (
            cv2.perspectiveTransform(
                source_box,
                image_to_roi,
            ).reshape(-1, 2)
        )


        objects.append(
            {
                "class_name": class_name,

                "confidence": float(score),

                "roi_center_px": (
                    roi_centre.tolist()
                ),

                "roi_polygon_px": (
                    roi_polygon.tolist()
                ),

                "selected_shelf_y_mm": (
                    float(
                        selected_shelf_y_mm
                    )
                ),

                # Estimated physical centre of the object.
                "coordinates_robot_mm": (
                    centre_point.tolist()
                ),

                # Physical point on the measured shelf.
                "shelf_contact_coordinates_robot_mm": (
                    contact_point.tolist()
                ),
            }
        )


objects.sort(
    key=lambda item: (
        item["selected_shelf_y_mm"],
        item["coordinates_robot_mm"][2],
    )
)


annotated = roi.copy()


for number, item in enumerate(
    objects,
    start=1,
):
    item["detection_id"] = number

    polygon = np.rint(
        item["roi_polygon_px"]
    ).astype(np.int32)

    roi_x, roi_y = (
        item["roi_center_px"]
    )

    centre_x, centre_y, centre_z = (
        item["coordinates_robot_mm"]
    )

    contact_x, contact_y, contact_z = (
        item[
            "shelf_contact_coordinates_robot_mm"
        ]
    )


    cv2.polylines(
        annotated,
        [polygon],
        True,
        (0, 200, 0),
        2,
    )


    cv2.circle(
        annotated,
        (
            int(round(roi_x)),
            int(round(roi_y)),
        ),
        5,
        (0, 0, 255),
        -1,
    )


    label = (
        f"#{number} "
        f"{item['class_name']} "
        f"C=({centre_x:.0f},"
        f"{centre_y:.0f},"
        f"{centre_z:.0f}) "
        f"P=({contact_x:.0f},"
        f"{contact_y:.0f},"
        f"{contact_z:.0f})"
    )


    label_x = max(
        0,
        int(
            polygon[:, 0].min()
        ),
    )

    label_y = max(
        28,
        int(
            polygon[:, 1].min()
        ) - 8,
    )


    cv2.putText(
        annotated,
        label,
        (
            label_x,
            label_y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 200, 0),
        2,
        cv2.LINE_AA,
    )


ANNOTATED_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

if not cv2.imwrite(
    str(ANNOTATED_OUTPUT),
    annotated,
):
    raise RuntimeError(
        f"Could not save annotated ROI: "
        f"{ANNOTATED_OUTPUT}"
    )


result = {
    "mode": (
        "single_camera_known_shelf_geometry"
    ),

    "units": "mm",

    "camera_position_robot_mm": (
        camera_position.tolist()
    ),

    "shelf_surface_y_mm": list(
        SHELF_SURFACE_Y_MM
    ),

    "roi_size_px": {
        "width": roi_width_px,
        "height": roi_height_px,
    },

    "explanation": {
        "coordinates_robot_mm": (
            "Estimated object centre. "
            "The X depth is calculated using "
            "the measured horizontal shelf."
        ),

        "shelf_contact_coordinates_robot_mm": (
            "Physical contact point on the "
            "measured shelf plane."
        ),
    },

    "objects": objects,
}


JSON_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

JSON_OUTPUT.write_text(
    json.dumps(
        result,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)


print(
    "Camera XYZ:",
    np.round(
        camera_position,
        2,
    ).tolist(),
)


for item in objects:
    centre = np.round(
        item["coordinates_robot_mm"],
        2,
    ).tolist()

    contact = np.round(
        item[
            "shelf_contact_coordinates_robot_mm"
        ],
        2,
    ).tolist()

    print(
        f"#{item['detection_id']} "
        f"{item['class_name']} "
        f"shelfY="
        f"{item['selected_shelf_y_mm']:.1f} "
        f"centerXYZ={centre} "
        f"contactXYZ={contact}"
    )


print(
    f"Full ROI image: {ROI_OUTPUT}"
)

print(
    f"Annotated ROI image: "
    f"{ANNOTATED_OUTPUT}"
)

print(
    f"Coordinates JSON: "
    f"{JSON_OUTPUT}"
)