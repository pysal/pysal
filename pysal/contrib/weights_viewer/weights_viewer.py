import wx
import pysal
from transforms import WorldToViewTransform

__author__ = "Charles R Schmidt <schmidtc@gmail.com>"

POINT_RADIUS = 5
BORDER_COLOR = wx.Colour(0,0,0,255)
SELECTION_COLOR = wx.Colour(255,128,0,255)
NEIGHBORS_COLOR = wx.Colour(128,255,0,255)
BACKGROUND_COLOR = wx.Colour(0,0,0,0)

class WeightsMapFrame(wx.Frame):
    def __init__(self,parent=None,size=(600,600), style=wx.DEFAULT_FRAME_STYLE, geo=None, w=None):
        wx.Frame.__init__(self,parent,size=size,style=style)
        self.Bind
        self.SetTitle("Weights Inspector")
        if issubclass(type(geo),basestring):
            geo = pysal.open(geo,'r')
        self.geo = geo
        if issubclass(type(w),basestring):
            w = pysal.open(w,'r').read()
        self.w = w
        self.wm = WeightsMap(self,self.geo,self.w)

class WeightsMap(wx.Panel):
    """ Display a Weights Inspection Map """
    def __init__(self, parent, geo, w_obj):
        wx.Panel.__init__(self,parent,size=(600,600))
        self.status = parent.CreateStatusBar(3)
        self.status.SetStatusWidths([-1,-2,-2])
        self.status.SetStatusText('No Selection',0)
        self.Bind(wx.EVT_SIZE, self.onSize)
        self.Bind(wx.EVT_PAINT, self.onPaint)
        self.Bind(wx.EVT_MOUSE_EVENTS, self.onMouse)
        self.trns = 0
        self.background = (255,255,255,255)
        w,h = self.GetSize()
        self.buffer = wx.EmptyBitmapRGBA(w,h,alpha=self.trns)
        self.geo = geo
        if geo.type == pysal.cg.shapes.Polygon:
            self.drawfunc = self.drawpoly
            self.locator = pysal.cg.PolygonLocator(geo)
        elif geo.type == pysal.cg.shapes.Point:
            self.drawfunc = self.drawpt
            self.locator = pysal.cg.PointLocator(geo)
        else:
            raise TypeError, "Unsupported Type: %r"%(geo.type)
        self.w = w_obj
        self._ids = range(self.w.n)
        self.transform = WorldToViewTransform(geo.bbox,w,h)
        self.selection = None
    def onMouse(self, evt):
        x,y = evt.X,evt.Y
        X,Y = self.transform.pixel_to_world(x,y)
        if self.geo.type == pysal.cg.shapes.Polygon:
            rs = self.locator.contains_point((X,Y))
            if rs:
                selection = rs[0].id-1
                self.set_selection(selection)
            else:
                self.set_selection(None)
        else:
            print self.locator.nearest((X,Y))
    def onSize(self, evt):
        w,h = self.GetSize()
        self.buffer = wx.EmptyBitmapRGBA(w,h,alpha=self.trns)
        self.transform = WorldToViewTransform(self.geo.bbox,w,h)
        self.draw()
    def onPaint(self, evt):
        pdc = wx.PaintDC(self)
        pdc.Clear()
        self.draw()
    def draw(self):
        dc = wx.MemoryDC()
        dc.SelectObject(self.buffer)
        dc.SetBackground(wx.Brush(wx.Colour(*self.background)))
        dc.Clear()
        dc.SelectObject(wx.NullBitmap)
        self.draw_shps(self.buffer)
        cdc = wx.ClientDC(self)
        cdc.DrawBitmap(self.buffer,0,0)
    def drawpoly(self,gc,matrix,fill=False,ids=None):
        geo = self.geo
        pth = gc.CreatePath()
        if not ids:
            ids = xrange(len(geo))
        for i in ids:
            poly = geo.get(i)
            parts = poly.parts
            if poly.holes[0]:
                parts = parts+poly.holes
            for part in parts:
                x,y = part[0]
                pth.MoveToPoint(x,y)
                for x,y in part[1:]:
                    pth.AddLineToPoint(x,y)
                pth.CloseSubpath()
        pth.Transform(matrix)
        if fill:
            gc.FillPath(pth)
        gc.StrokePath(pth)
        return gc
    def drawpt(self,gc,matrix,fill=False,ids=None):
        r = POINT_RADIUS
        radius = r/2.0
        geo = self.geo
        for pt in geo:
            x,y = matrix.TransformPoint(*pt)
            gc.DrawEllipse(x-radius,y-radius,r,r)
        return gc
    def draw_shps(self, buff, fill_color=None, ids=None, fill_style=wx.SOLID):
        transform = self.transform
        dc = wx.MemoryDC()
        dc.SelectObject(buff)
        gc = wx.GraphicsContext.Create(dc)
        gc.SetPen( gc.CreatePen(wx.Pen(BORDER_COLOR,1)) )
        if fill_color != None:
            gc.SetBrush( gc.CreateBrush(wx.Brush(fill_color,fill_style)) )
            fill = True
        else:
            fill = False
        matrix = gc.CreateMatrix()
        matrix.Scale(1./transform.scale,1./-transform.scale)
        matrix.Translate(*transform.offset)
        self.drawfunc(gc,matrix,fill,ids)
    def set_selection(self,sel):
        if self.selection == sel:
            return
        self.selection = sel

        cdc = wx.ClientDC(self)
        if sel != None:
            w,h = self.transform.pixel_size
            buff = self.buffer.GetSubBitmap((0,0,w,h))
            id = self.w.id_order[sel]
            neighbors = map(self.w.id_order.index,self.w.neighbors[id])
            self.draw_shps(buff, NEIGHBORS_COLOR, neighbors)
            if sel in neighbors:
                self.draw_shps(buff, SELECTION_COLOR, [sel], wx.CROSSDIAG_HATCH)
            else:
                self.draw_shps(buff, SELECTION_COLOR, [sel])
            print sel,":",neighbors
            cdc.DrawBitmap(buff,0,0)
            stat0 = "Selection:%s"%id
            stat1 = "Neighbors:%s"%(','.join(map(str,self.w.neighbors[id])))
            stat2 = "Weights:%s"%(','.join(map(str,self.w.weights[id])))
            self.status.SetStatusText(stat0,0)
            self.status.SetStatusText(stat1,1)
            self.status.SetStatusText(stat2,2)
        else:
            self.status.SetStatusText('No Selection',0)
            self.status.SetStatusText('',1)
            self.status.SetStatusText('',2)
            cdc.DrawBitmap(self.buffer,0,0)

class WeightsMapApp(wx.App):
    def __init__(self, geo=None, w=None, redirect=False):
        self.geo = geo
        self.w = w
        wx.App.__init__(self, redirect)
    def OnInit(self):
        self.SetAppName("Weights Inspector")
        self.frame = WeightsMapFrame(None,size=(600,600),geo=self.geo,w=self.w)
        self.SetTopWindow(self.frame)
        self.frame.Show()
        return True


if __name__=='__main__':
    #shp = pysal.examples.get_path('sids2.shp')
    #w = pysal.examples.get_path('sids2.gal')
    #app = WeightsMapApp(shp,w)
    #app.MainLoop()

    shp = pysal.examples.get_path('columbus.shp')
    w = pysal.queen_from_shapefile(shp)
    app = WeightsMapApp(shp,w)
    app.MainLoop()
