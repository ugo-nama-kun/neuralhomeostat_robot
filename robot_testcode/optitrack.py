import natnet

IP_OPTITRACK = "192.168.1.36"

# OptiTrack server
optitrack_client = natnet.Client.connect(server=IP_OPTITRACK)

object_measurement = None
object_names = {
    1: "ANT",
    2: "FOOD",
}

def callback(rigid_bodies, markers, timing):
    global object_measurement, object_names

    object_measurement = {
        object_names[rb.id_]: (rb.position, rb.orientation) for rb in rigid_bodies
    }
    
    print(object_measurement)


optitrack_client.set_callback(callback)
optitrack_client.spin()
