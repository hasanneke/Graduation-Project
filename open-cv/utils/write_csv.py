def write_csv(results, output_path):
    with open(output_path, "w") as f:
        f.write(
            "{},{},{},{},{},{},{}\n".format(
                "frame_nmr",
                "car_id",
                "car_bbox",
                "license_plate_bbox",
                "license_plate_bbox_score",
                "license_number",
                "license_number_score",
            )
        )

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if (
                    "car" in results[frame_nmr][car_id].keys()
                    and "license_plate" in results[frame_nmr][car_id].keys()
                    and "text" in results[frame_nmr][car_id]["license_plate"].keys()
                ):
                    f.write(
                        "{},{},{},{},{},{},{}\n".format(
                            frame_nmr,
                            car_id,
                            "[{} {} {} {}]".format(
                                results[frame_nmr][car_id]["car"]["bbox"][0],
                                results[frame_nmr][car_id]["car"]["bbox"][1],
                                results[frame_nmr][car_id]["car"]["bbox"][2],
                                results[frame_nmr][car_id]["car"]["bbox"][3],
                            ),
                            "[{} {} {} {}]".format(
                                results[frame_nmr][car_id]["license_plate"]["bbox"][0],
                                results[frame_nmr][car_id]["license_plate"]["bbox"][1],
                                results[frame_nmr][car_id]["license_plate"]["bbox"][2],
                                results[frame_nmr][car_id]["license_plate"]["bbox"][3],
                            ),
                            results[frame_nmr][car_id]["license_plate"]["bbox_score"],
                            results[frame_nmr][car_id]["license_plate"]["text"],
                            results[frame_nmr][car_id]["license_plate"]["text_score"],
                        )
                    )
        f.close()
