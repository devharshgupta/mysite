



// ******************************
const form = document.querySelector('.form');

let mapZoomLevel = 15
let map

if('geolocation' in navigator) {
  navigator.geolocation.getCurrentPosition((position) => {
    test = [position.coords.latitude,position.coords.longitude]
    console.log(test);



 map = L.map('map').setView(test, mapZoomLevel);

L.tileLayer('https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', {
  attribution:
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}).addTo(map);

let marker = L.marker(test).addTo(map);


map.on("contextmenu", function (event) {

  
  document.getElementById('lat').value = event.latlng.lat
  document.getElementById('long').value = event.latlng.lng

  L.marker(event.latlng).addTo(map);
});

});
} else{
  alert("Please allow map to fetch your location")

  map = L.map('map').setView([25.4582113,78.5751338], mapZoomLevel);

  L.tileLayer('https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', {
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map)

}