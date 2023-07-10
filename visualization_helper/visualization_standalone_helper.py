import numpy as np
import vtk
import os 
import sys
import seaborn as sns
import random
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import pickle

try:
    from plyfile import PlyData
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)

ScanNet_OBJ_CLASS_IDS = np.array([ 1,  7,  8, 13, 20, 31, 34, 43])
ShapeNetIDMap = {'4379243': 'table', '3593526': 'jar', '4225987': 'skateboard', '2958343': 'car', '2876657': 'bottle', '4460130': 'tower', '3001627': 'chair', '2871439': 'bookshelf', '2942699': 'camera', '2691156': 'airplane', '3642806': 'laptop', '2801938': 'basket', '4256520': 'sofa', '3624134': 'knife', '2946921': 'can', '4090263': 'rifle', '4468005': 'train', '3938244': 'pillow', '3636649': 'lamp', '2747177': 'trash_bin', '3710193': 'mailbox', '4530566': 'watercraft', '3790512': 'motorbike', '3207941': 'dishwasher', '2828884': 'bench', '3948459': 'pistol', '4099429': 'rocket', '3691459': 'loudspeaker', '3337140': 'file cabinet', '2773838': 'bag', '2933112': 'cabinet', '2818832': 'bed', '2843684': 'birdhouse', '3211117': 'display', '3928116': 'piano', '3261776': 'earphone', '4401088': 'telephone', '4330267': 'stove', '3759954': 'microphone', '2924116': 'bus', '3797390': 'mug', '4074963': 'remote', '2808440': 'bathtub', '2880940': 'bowl', '3085013': 'keyboard', '3467517': 'guitar', '4554684': 'washer', '2834778': 'bicycle', '3325088': 'faucet', '4004475': 'printer', '2954340': 'cap', '3046257': 'clock', '3513137': 'helmet', '3991062': 'flowerpot', '3761084': 'microwaves'}

class Vis_base(object):
    '''
        Visualization class for scannet frames.
    '''

    def __init__(self, scene_points, instance_models, center_list, vector_list):
        self.scene_points = scene_points
        self.instance_models = instance_models
        self.cam_K = np.array([[300, 0, 600], [0, 500, 400], [0, 0, 1]])
        self.center_list = center_list
        self.vector_list = vector_list
        self.palette_cls = np.array([*sns.color_palette("hls", len(ScanNet_OBJ_CLASS_IDS))])
        self.depth_palette = np.array(sns.color_palette("crest_r", n_colors=100))
        self.palette_inst = np.array([*sns.color_palette("hls", 10)])

    def get_box_corners(self, center, vectors):
        '''
        Convert box center and vectors to the corner-form
        :param center:
        :param vectors:
        :return: corner points and faces related to the box
        '''
        corner_pnts = [None] * 8
        corner_pnts[0] = tuple(center - vectors[0] - vectors[1] - vectors[2])
        corner_pnts[1] = tuple(center + vectors[0] - vectors[1] - vectors[2])
        corner_pnts[2] = tuple(center + vectors[0] + vectors[1] - vectors[2])
        corner_pnts[3] = tuple(center - vectors[0] + vectors[1] - vectors[2])

        corner_pnts[4] = tuple(center - vectors[0] - vectors[1] + vectors[2])
        corner_pnts[5] = tuple(center + vectors[0] - vectors[1] + vectors[2])
        corner_pnts[6] = tuple(center + vectors[0] + vectors[1] + vectors[2])
        corner_pnts[7] = tuple(center - vectors[0] + vectors[1] + vectors[2])

        faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (0, 4, 7, 3)]

        return corner_pnts, faces

    def set_mapper(self, prop, mode):
        mapper = vtk.vtkPolyDataMapper()

        if mode == 'model':
            mapper.SetInputConnection(prop.GetOutputPort())

        elif mode == 'box':
            if vtk.VTK_MAJOR_VERSION <= 5:
                mapper.SetInput(prop)
            else:
                mapper.SetInputData(prop)

        else:
            raise IOError('No Mapper mode found.')

        return mapper
    
    def set_actor(self, mapper):
        '''
        vtk general actor
        :param mapper: vtk shape mapper
        :return: vtk actor
        '''
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        return actor

    def set_bbox_line_actor(self, corners, faces, color):
        edge_set1 = np.vstack([np.array(faces)[:, 0], np.array(faces)[:, 1]]).T
        edge_set2 = np.vstack([np.array(faces)[:, 1], np.array(faces)[:, 2]]).T
        edge_set3 = np.vstack([np.array(faces)[:, 2], np.array(faces)[:, 3]]).T
        edge_set4 = np.vstack([np.array(faces)[:, 3], np.array(faces)[:, 0]]).T
        edges = np.vstack([edge_set1, edge_set2, edge_set3, edge_set4])
        edges = np.unique(np.sort(edges, axis=1), axis=0)

        pts = vtk.vtkPoints()
        for corner in corners:
            pts.InsertNextPoint(corner)

        lines = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        for edge in edges:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, edge[0])
            line.GetPointIds().SetId(1, edge[1])
            lines.InsertNextCell(line)
            colors.InsertNextTuple3(*color)

        linesPolyData = vtk.vtkPolyData()
        linesPolyData.SetPoints(pts)
        linesPolyData.SetLines(lines)
        linesPolyData.GetCellData().SetScalars(colors)

        return linesPolyData

    def get_bbox_line_actor(self, center, vectors, color, opacity, width=10):
        corners, faces = self.get_box_corners(center, vectors)
        bbox_actor = self.set_actor(self.set_mapper(self.set_bbox_line_actor(corners, faces, color), 'box'))
        bbox_actor.GetProperty().SetOpacity(opacity)
        bbox_actor.GetProperty().SetLineWidth(width)
        return bbox_actor


    def set_arrow_actor(self, startpoint, vector):
        '''
        Design an actor to draw an arrow from startpoint to startpoint + vector.
        :param startpoint: 3D point
        :param vector: 3D vector
        :return: an vtk arrow actor
        '''
        arrow_source = vtk.vtkArrowSource()
        arrow_source.SetTipLength(0.2)
        arrow_source.SetTipRadius(0.08)
        arrow_source.SetShaftRadius(0.02)

        vector = vector / np.linalg.norm(vector) * 0.5

        endpoint = startpoint + vector

        # compute a basis
        normalisedX = [0 for i in range(3)]
        normalisedY = [0 for i in range(3)]
        normalisedZ = [0 for i in range(3)]

        # the X axis is a vector from start to end
        math = vtk.vtkMath()
        math.Subtract(endpoint, startpoint, normalisedX)
        length = math.Norm(normalisedX)
        math.Normalize(normalisedX)

        # the Z axis is an arbitrary vector cross X
        arbitrary = [0 for i in range(3)]
        arbitrary[0] = random.uniform(-10, 10)
        arbitrary[1] = random.uniform(-10, 10)
        arbitrary[2] = random.uniform(-10, 10)
        math.Cross(normalisedX, arbitrary, normalisedZ)
        math.Normalize(normalisedZ)

        # the Y axis is Z cross X
        math.Cross(normalisedZ, normalisedX, normalisedY)

        # create the direction cosine matrix
        matrix = vtk.vtkMatrix4x4()
        matrix.Identity()
        for i in range(3):
            matrix.SetElement(i, 0, normalisedX[i])
            matrix.SetElement(i, 1, normalisedY[i])
            matrix.SetElement(i, 2, normalisedZ[i])

        # apply the transform
        transform = vtk.vtkTransform()
        transform.Translate(startpoint)
        transform.Concatenate(matrix)
        transform.Scale(length, length, length)

        # create a mapper and an actor for the arrow
        mapper = vtk.vtkPolyDataMapper()
        actor = vtk.vtkActor()

        mapper.SetInputConnection(arrow_source.GetOutputPort())
        actor.SetUserMatrix(transform.GetMatrix())
        actor.SetMapper(mapper)

        return actor
    
    def set_camera(self, position, focal_point, cam_K):
        camera = vtk.vtkCamera()
        camera.SetPosition(*position)
        camera.SetFocalPoint(*focal_point[0])
        camera.SetViewUp(*focal_point[1])
        camera.SetViewAngle((2*np.arctan(cam_K[1][2]/cam_K[0][0]))/np.pi*180)
        return camera
    
    def set_points_property(self, point_clouds, point_colors):
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName('Color')

        x3 = point_clouds[:, 0]
        y3 = point_clouds[:, 1]
        z3 = point_clouds[:, 2]

        for x, y, z, c in zip(x3, y3, z3, point_colors):
            id = points.InsertNextPoint([x, y, z])
            colors.InsertNextTuple3(*c)
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(id)

        # Create a polydata object
        point = vtk.vtkPolyData()
        # Set the points and vertices we created as the geometry and topology of the polydata
        point.SetPoints(points)
        point.SetVerts(vertices)
        point.GetPointData().SetScalars(colors)
        point.GetPointData().SetActiveScalars('Color')

        return point
    
    def create_color_map(self, num_colors):
        colors = []
        for _ in range(num_colors):
            r, g, b = random.randint(0,255), random.randint(0,255), random.randint(0,255) 
            colors.append((r,g,b))
        return colors

    def random_sampling(self, pc, num_sample, replace=None, return_choices=False):
        """ Input is NxC, output is num_samplexC
        """
        if replace is None: replace = (pc.shape[0] < num_sample)
        choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
        if return_choices:
            return pc[choices], choices
        else:
            return pc[choices]

    def set_render(self, centroid, only_points, min_max_dist, label_type=None, *args, **kwargs):
        renderer = vtk.vtkRenderer()
        renderer.ResetCamera()

        # '''draw world system'''
        # renderer.AddActor(self.set_axes_actor())

        '''set camera'''
        camera = self.set_camera(centroid, [[0., 0., 0.], [-centroid[0], -centroid[1], centroid[0]**2/centroid[2] + centroid[1]**2/centroid[2]]], self.cam_K)
        renderer.SetActiveCamera(camera)

        '''draw scene points'''
        point_size = 4

        if label_type and label_type != 'custom_boxes':
            boxes3D, point_cloud, point_instance_labels, semantic_labels = fetch_custom_data() 

            # padding centers with a large number
            # compute GT Centers *AFTER* augmentation
            # generate gt centers
            # Note: since there's no map between bbox instance labels and
            # pc instance_labels (it had been filtered 
            # in the data preparation step) we'll compute the instance bbox
            # from the points sharing the same instance label.
            target_bboxes = np.zeros((64, 6))
            target_bboxes[0:boxes3D.shape[0], :] = boxes3D[:,0:6]

            gt_centers = target_bboxes[:, 0:3]
            gt_centers[boxes3D.shape[0]:, :] += 1000.0  

            point_obj_mask = np.zeros(len(point_cloud))
            point_instance_label = np.zeros(len(point_cloud)) - 1

            for i_instance in np.unique(point_instance_labels):
                # find all points belong to that instance
                ind = np.where(point_instance_labels == i_instance)[0]

                # find the semantic label
                if semantic_labels[ind[0]] in ScanNet_OBJ_CLASS_IDS:
                    x = point_cloud[ind, :3]
                    center = 0.5 * (x.min(0) + x.max(0))
                    ilabel = np.argmin(((center - gt_centers) ** 2).sum(-1))
                    point_instance_label[ind] = ilabel
                    point_obj_mask[ind] = 1.0

            masked_point_cloud = []
            masked_point_instance_labels = []
            masked_semantic_labels = []
            for idx, point in enumerate(point_cloud):
                if point_obj_mask[idx] == 1:
                    masked_point_cloud.append(point)
                    masked_point_instance_labels.append(point_instance_label[idx])
                    masked_semantic_labels.append(semantic_labels[idx])
                    
            if label_type == 'point_instance_labels':
                labels = point_instance_label.astype(np.int64)
            elif label_type == 'masked_point_instance_labels':
                point_cloud = np.array(masked_point_cloud)
                labels  = np.array(masked_point_instance_labels).astype(int)
            elif label_type == 'point_semantic_instance_labels':
                labels = semantic_labels
            elif label_type == 'masked_point_semantic_instance_labels':
                point_cloud = np.array(masked_point_cloud)
                labels = np.array(masked_semantic_labels).astype(int)
            
            color_map = self.create_color_map(np.max(labels) + 1)
            point_cloud, choices = self.random_sampling(point_cloud, self.scene_points, return_choices=True)
            labels = labels[choices]      
            colors = []
            for label in labels:
                colors.append(color_map[label])
            colors = np.array(colors)
        else:
            point_cloud = self.scene_points
            colors = np.linalg.norm(point_cloud[:, :3]-centroid, axis=1)
            colors = self.depth_palette[np.int16((colors-colors.min())/(colors.max()-colors.min())*99)]
        
        point_actor = self.set_actor(
            self.set_mapper(self.set_points_property(point_cloud[:, :3], 255*colors), 'box')
        )
        point_actor.GetProperty().SetPointSize(point_size)
        point_actor.GetProperty().SetOpacity(0.3)
        point_actor.GetProperty().SetInterpolationToPBR()
        renderer.AddActor(point_actor)

        if not only_points:
            '''draw shapenet models'''
            dists = np.linalg.norm(np.array(self.center_list)-centroid, axis=1)
            if min_max_dist is None:
                min_max_dist = [min(dists), max(dists)]
            dists = (dists - min_max_dist[0])/(min_max_dist[1]-min_max_dist[0])
            dists = np.clip(dists, 0, 1)
            inst_color_ids = np.round(dists*(self.palette_inst.shape[0]-1)).astype(np.uint8)

            for obj, color_id in zip(self.instance_models, inst_color_ids):
                object_actor = self.set_actor(self.set_mapper(obj, 'model'))
                object_actor.GetProperty().SetColor(self.palette_inst[color_id])
                object_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(object_actor)

            '''draw bounding boxes'''
            for center, vectors in zip(self.center_list, self.vector_list):
                box_line_actor = self.get_bbox_line_actor(center, vectors, [64, 64, 64], 1., 3)
                box_line_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(box_line_actor)

                # draw orientations
                color = [[1, 0, 0], [0, 1, 0], [0., 0., 1.]]

                for index in range(vectors.shape[0]):
                    arrow_actor = self.set_arrow_actor(center, vectors[index])
                    arrow_actor.GetProperty().SetColor(color[index])
                    renderer.AddActor(arrow_actor)

        '''light'''
        positions = [(10, 10, 10), (-10, 10, 10), (10, -10, 10), (-10, -10, 10)]
        for position in positions:
            light = vtk.vtkLight()
            light.SetIntensity(1.5)
            light.SetPosition(*position)
            light.SetPositional(True)
            light.SetFocalPoint(0, 0, 0)
            light.SetColor(1., 1., 1.)
            renderer.AddLight(light)

        renderer.SetBackground(1., 1., 1.)
        return renderer, min_max_dist

    def set_render_window(self, centroid, only_points, min_max_dist, offline, label_type=None):
        render_window = vtk.vtkRenderWindow()
        renderer, min_max_dist = self.set_render(centroid, only_points, min_max_dist, label_type=label_type)
        renderer.SetUseDepthPeeling(1)
        render_window.AddRenderer(renderer)
        render_window.SetSize(*np.int32((self.cam_K[:2,2]*2)))
        render_window.SetOffScreenRendering(offline)

        return render_window, min_max_dist

    def visualize(self, centroid=np.array([0, -2.5, 2.5]), save_path = None, only_points=False, offline=False, min_max_dist=None, label_type=None):
        '''
        Visualize a 3D scene.
        '''
        render_window, min_max_dist = self.set_render_window(centroid, only_points, min_max_dist, offline, label_type=label_type)
        render_window.Render()

        if save_path is not None:
            windowToImageFilter = vtk.vtkWindowToImageFilter()
            windowToImageFilter.SetInput(render_window)
            windowToImageFilter.Update()

            writer = vtk.vtkPNGWriter()
            writer.SetFileName(save_path)
            writer.SetInputConnection(windowToImageFilter.GetOutputPort())
            writer.Write()

        if not offline:
            render_window_interactor = vtk.vtkRenderWindowInteractor()
            render_window_interactor.SetRenderWindow(render_window)
            render_window_interactor.Start()
        return min_max_dist 

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

def angle2class(angle):
    ''' Convert continuous angle to discrete class
        [optinal] also small regression number from
        class center angle to current angle.
        angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        return is class of int32 of 0,1,...,N-1 and a number such that
            class*(2pi/N) + number = angle
    '''
    num_class = 12
    angle = angle % (2 * np.pi)
    assert False not in (angle >= 0) * (angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_class)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = np.int16(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle

def fetch_custom_data():
    with open(f'bbox.pkl', 'rb') as file:
        box_info = pickle.load(file)
        boxes3D = []
        for item in box_info:
            boxes3D.append(item['box3D'])
        boxes3D = np.array(boxes3D)
        scan_data = np.load(f'full_scan.npz')
        point_cloud = scan_data['mesh_vertices']
        point_instance_labels = scan_data['instance_labels']
        semantic_labels = scan_data['semantic_labels']
    return boxes3D, point_cloud, point_instance_labels, semantic_labels

def visualize(output_dir, offline, label_type=None, only_points=False):
    predicted_boxes = np.load(os.path.join(output_dir, '000000_pred_confident_nms_bbox.npz'))
    input_point_cloud = read_ply(os.path.join(output_dir, '000000_pc.ply'))
    bbox_params = predicted_boxes['obbs']
    '''
    Load custom data into the visualization_helper. Just add your custom `bbox.pkl` 
    and `full_scan.npz` to the folder in which the visualization helper is installed.
    '''
    if label_type == 'custom_boxes':
        boxes3D, _, _, _ = fetch_custom_data()
        bbox_params = boxes3D
    
    proposal_map = predicted_boxes['proposal_map']
    transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    instance_models = []
    center_list = []
    vector_list = []
    for map_data, bbox_param in zip(proposal_map, bbox_params):
        mesh_file = os.path.join(output_dir, 'proposal_%d_mesh.ply' % tuple(map_data))
        ply_reader = vtk.vtkPLYReader()
        ply_reader.SetFileName(mesh_file)
        ply_reader.Update()
        # get points from object
        polydata = ply_reader.GetOutput()
        # read points using vtk_to_numpy
        obj_points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float)
        '''Fit obj points to bbox'''
        center = bbox_param[:3]
        orientation = bbox_param[6]
        sizes = bbox_param[3:6]
        obj_points = obj_points - (obj_points.max(0) + obj_points.min(0))/2.
        obj_points = obj_points.dot(transform_m.T)
        obj_points = obj_points.dot(np.diag(1/(obj_points.max(0) - obj_points.min(0)))).dot(np.diag(sizes))
        axis_rectified = np.array([[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
        obj_points = obj_points.dot(axis_rectified) + center
        points_array = numpy_to_vtk(obj_points[..., :3], deep=True)
        polydata.GetPoints().SetData(points_array)
        ply_reader.Update()
        '''draw bboxes'''
        vectors = np.diag(sizes/2.).dot(axis_rectified)
        instance_models.append(ply_reader)
        center_list.append(center)
        vector_list.append(vectors)
    scene = Vis_base(scene_points=input_point_cloud, instance_models=instance_models, center_list=center_list,
                     vector_list=vector_list)
    camera_center = np.array([0, -3, 3])
    scene.visualize(centroid=camera_center, offline=offline, save_path=os.path.join(output_dir, 'pred.png'), label_type=label_type, only_points=only_points)

visualize("demo_results", offline=False, label_type=None, only_points=False)