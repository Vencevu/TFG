from spade import agent, quit_spade
import carla
import random
import time
import open3d as o3d

#Clase Agente
class DummyAgent(agent.Agent):
    
    actor_list = []
    world = None
    client = None

    def save_lidar(self, image):
        print(5)

    async def setup(self):
        #Conexion con CARLA
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(10.0)

        #Configuracion del mundo
        self.world = self.client.get_world()

        #Creamos vehiculo
        blueprint_library = self.world.get_blueprint_library()
        bp = random.choice(blueprint_library.filter('vehicle'))
        transform = random.choice(self.world.get_map().get_spawn_points()) 
        vehicle = self.world.spawn_actor(bp, transform) 
        self.actor_list.append(vehicle)

        #Creamos sensor y lo acoplamos al vehiculo
        lidar_cam = None
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('dropoff_general_rate', lidar_bp.get_attribute('dropoff_general_rate').recommended_values[0])
        lidar_bp.set_attribute('dropoff_intensity_limit', lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
        lidar_bp.set_attribute('dropoff_zero_intensity', lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])
        lidar_bp.set_attribute('channels',"64")
        lidar_bp.set_attribute('points_per_second',"100000")
        lidar_bp.set_attribute('rotation_frequency',"20")
        lidar_bp.set_attribute('range',"50")
        lidar_location = carla.Location(0,0,2)
        lidar_rotation = carla.Rotation(0,0,0)
        lidar_transform = carla.Transform(lidar_location,lidar_rotation)
        lidar_sen = self.world.spawn_actor(lidar_bp,lidar_transform,attach_to=vehicle)
        #lidar_sen.listen(lambda point_cloud: point_cloud.save_to_disk('test_images/%.6d.ply' % point_cloud.frame))
        lidar_sen.listen(self.save_lidar)

        self.actor_list.append(lidar_sen)


        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        

#Lanzamos el agente
dummy = DummyAgent("agente@localhost", "1234")

future = dummy.start()
future.result()


# try:
#     while True:
#         time.sleep(1)
# except KeyboardInterrupt:
#     dummy.stop()

dummy.stop()
quit_spade()
